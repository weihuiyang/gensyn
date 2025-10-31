import os
import time
import logging
from collections import defaultdict
from genrl.blockchain import SwarmCoordinator
from genrl.communication import Communication
from genrl.communication.hivemind.hivemind_backend import HivemindBackend
from genrl.data import DataManager
from genrl.game import BaseGameManager
from genrl.game.game_manager import DefaultGameManagerMixin
from genrl.logging_utils.global_defs import get_logger
from genrl.logging_utils.system_utils import get_system_info
from genrl.rewards import RewardManager
from genrl.roles import RoleManager
from genrl.state import GameState
from genrl.trainer import TrainerModule
from huggingface_hub import login, whoami
from rgym_exp.src.utils.name_utils import get_name_from_peer_id
from rgym_exp.src.prg_module import PRGModule
import threading
import requests


class SwarmGameManager(BaseGameManager, DefaultGameManagerMixin):
    """GameManager that orchestrates a game using a SwarmCoordinator."""

    def __init__(
        self,
        coordinator: SwarmCoordinator,
        max_stage: int,
        max_round: int,
        game_state: GameState,
        reward_manager: RewardManager,
        trainer: TrainerModule,
        data_manager: DataManager,
        communication: Communication,
        role_manager: RoleManager | None = None,
        run_mode: str = "train",
        log_dir: str = "logs",
        hf_token: str | None = None,
        hf_push_frequency: int = 20,
        submit_frequency: int = 3,
        **kwargs,
    ):
        super().__init__(
            max_stage=max_stage,
            max_round=max_round,
            game_state=game_state,
            reward_manager=reward_manager,
            trainer=trainer,
            data_manager=data_manager,
            communication=communication,
            role_manager=role_manager,
            run_mode=run_mode,
        )

        assert isinstance(self.communication, HivemindBackend)
        self.train_timeout = 60 * 60 * 24 * 31  # 1 month

        # Logging Setup
        self.peer_id = self.communication.get_id()
        self.state.peer_id = self.peer_id
        self.animal_name = get_name_from_peer_id(self.peer_id, True)

        format_msg = f"[{self.animal_name}] %(asctime)s %(levelname)s: %(message)s"
        logging.basicConfig(level=logging.INFO, format=format_msg)
        formatter = logging.Formatter(format_msg)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"training_{self.animal_name}.log")
        )
        file_handler.setFormatter(formatter)
        _LOG = get_logger()
        _LOG.addHandler(file_handler)

        # Register peer_id and get current round from the chain
        self.coordinator = coordinator
        self.coordinator.register_peer(self.peer_id)
        round, _ = self.coordinator.get_round_and_stage()
        self.state.round = round
        self.communication.step_ = self.state.round  # initialize communication module to contract's round

        self.submit_frequency = submit_frequency
        self.batched_signals = 0.0
        self.submitted_this_round = False
        self.cached_zero_reward_rounds = []  # zero reward cache

        # PRG Game
        self.prg_module = PRGModule(log_dir, **kwargs)
        self.prg_game = self.prg_module.prg_game

        # enable push to HF if token was provided
        self.hf_token = hf_token
        if self.hf_token not in [None, "None"]:
            username = whoami(token=self.hf_token)["name"]
            model_name = self.trainer.model.config.name_or_path.split("/")[-1]
            model_name += "-Gensyn-Swarm"
            model_name += f"-{self.animal_name}"
            self.trainer.args.hub_model_id = f"{username}/{model_name}"
            self.trainer.args.push_to_hub = True
            self.trainer.args.hub_token = self.hf_token
            self.hf_push_frequency = hf_push_frequency
            get_logger().info("Logging into Hugging Face Hub...")
            login(self.hf_token)

        get_logger().info(
            f"🐱 Hello 🐈 [{get_name_from_peer_id(self.peer_id)}] 🦮 [{self.peer_id}]!"
        )
        get_logger().info(f"bootnodes: {kwargs.get('bootnodes', [])}")
        get_logger().info(f"Using Model: {self.trainer.model.config.name_or_path}")

        with open(os.path.join(log_dir, f"system_info.txt"), "w") as f:
            f.write(get_system_info())

    def _get_total_rewards_by_agent(self):
        rewards_by_agent = defaultdict(int)
        for stage in range(self.state.stage):
            rewards = self.rewards[stage]
            for agent_id, agent_rewards in rewards.items():
                for batch_id, batch_rewards in agent_rewards.items():
                    tot = 0
                    for generation_rewards in batch_rewards:
                        tot += sum(generation_rewards)
                    rewards_by_agent[agent_id] += tot
        return rewards_by_agent

    # ----------------- 保留原有接口 -----------------
    def _try_submit_to_chain(self, signal_by_agent):
        """原始方法保留，可以按小时提交使用"""
        elapsed_time_hours = (time.time() - getattr(self, "time_since_submit", 0)) / 3600
        if elapsed_time_hours > self.submit_frequency:
            try:
                self.coordinator.submit_reward(
                    self.state.round, 0, int(self.batched_signals), self.peer_id
                )
                self.batched_signals = 0.0
                if len(signal_by_agent) > 0:
                    max_agent, max_signal = max(
                        signal_by_agent.items(), key=lambda x: x[1]
                    )
                else:
                    max_agent = self.peer_id

                self.coordinator.submit_winners(
                    self.state.round, [max_agent], self.peer_id
                )
                self.time_since_submit = time.time()
                self.submitted_this_round = True
            except Exception as e:
                get_logger().debug(str(e))

    # ----------------- 新逻辑接口 -----------------
    def _hook_after_rewards_updated(self):
        """奖励更新后自动提交，整合 zero reward 缓存"""
        rewards_by_agent = self._get_total_rewards_by_agent()
        if not rewards_by_agent:
            get_logger().warning(f"No rewards data for round {self.state.round}")
            return

        my_rewards = rewards_by_agent.get(self.peer_id, 0)
        my_rewards = (my_rewards + 1) * (my_rewards > 0) + my_rewards * (my_rewards <= 0)

        submission_success = self._submit_current_rewards(my_rewards, rewards_by_agent)

        if submission_success:
            # zero reward 缓存逻辑
            if my_rewards <= 0:
                if self.state.round not in self.cached_zero_reward_rounds:
                    self.cached_zero_reward_rounds.append(self.state.round)
                if len(self.cached_zero_reward_rounds) > 3:
                    removed = self.cached_zero_reward_rounds.pop(0)
                    get_logger().info(f"Zero reward cache exceeded 3, removed round {removed}")
            else:
                # 清空缓存
                self.cached_zero_reward_rounds = []

    def _submit_current_rewards(self, my_rewards, rewards_by_agent):
        """提交奖励和胜者，包含缓存 zero reward"""
        try:
            max_agent, max_rewards = max(rewards_by_agent.items(), key=lambda x: x[1])
            total_cached_zero = len(self.cached_zero_reward_rounds)
            final_rewards = my_rewards + total_cached_zero

            get_logger().info(
                f"Submitting rewards: {final_rewards} (current: {my_rewards} + cached zero: {total_cached_zero}) for round {self.state.round}"
            )

            # 提交奖励（重试3次）
            for attempt in range(3):
                try:
                    self.coordinator.submit_reward(
                        self.state.round, 0, int(final_rewards), self.peer_id
                    )
                    get_logger().info(
                        f"Successfully submitted reward {int(final_rewards)} for round {self.state.round}"
                    )
                    break
                except Exception as e:
                    get_logger().warning(f"Submit reward attempt {attempt+1} failed: {e}")
                    if attempt == 2:
                        get_logger().error(f"Failed to submit reward after 3 attempts: {e}")
                        return False

            # 提交胜者（重试3次）
            for attempt in range(3):
                try:
                    self.coordinator.submit_winners(
                        self.state.round, [max_agent], self.peer_id
                    )
                    get_logger().info(
                        f"Successfully submitted winner {max_agent} (rewards: {max_rewards})"
                    )
                    break
                except Exception as e:
                    get_logger().warning(f"Submit winners attempt {attempt+1} failed: {e}")
                    if attempt == 2:
                        get_logger().error(f"Failed to submit winners after 3 attempts: {e}")
                        return False

            # 清理 zero reward 缓存
            if my_rewards > 0 or total_cached_zero > 0:
                self.cached_zero_reward_rounds = []

            self.submitted_this_round = True
            
            return True

        except Exception as e:
            get_logger().error(f"Error in _submit_current_rewards: {e}")
            return False

    def _hook_after_round_advanced(self):
        """轮次推进后，保存 HF 并阻塞等待下一轮"""
        if self.prg_game:
            prg_history_dict = self.prg_module.prg_history_dict
            results_dict = self.trainer.play_prg_game_logits(prg_history_dict)
            self.prg_module.play_prg_game(results_dict, self.peer_id)

        self._save_to_hf()
        self.agent_block()
        self.submitted_this_round = False

    def _hook_after_game(self):
        self._save_to_hf()

    def _save_to_hf(self):
        if (
            self.hf_token not in [None, "None"]
            and self.state.round % self.hf_push_frequency == 0
        ):
            get_logger().info(f"pushing model to huggingface")
            try:
                repo_id = self.trainer.args.hub_model_id
                self.trainer.model.push_to_hub(
                    repo_id=repo_id,
                    token=self.hf_token,
                    commit_message=f"rl-swarm: round {self.state.round}, agent {self.animal_name}",
                    tags=[
                        "rl-swarm",
                        "genrl-swarm",
                        "grpo",
                        "gensyn",
                        f"I am {self.animal_name}",
                    ],
                )
            except Exception:
                get_logger().exception(
                    "Failed to push model to the Hugging Face Hub.",
                    stack_info=True,
                )

    def agent_block(
        self, check_interval=5.0, log_timeout=10.0, max_check_interval=60.0 * 5
    ):
        start_time = time.monotonic()
        fetch_log_time = start_time
        check_backoff = check_interval
        while time.monotonic() - start_time < self.train_timeout:
            curr_time = time.monotonic()
            _ = self.communication.dht.get_visible_maddrs(latest=True)

            try:
                round_num, stage = self.coordinator.get_round_and_stage()
            except Exception as e:
                if curr_time - fetch_log_time > log_timeout:
                    get_logger().debug(
                        f"Could not fetch round and stage: {e}. Next check in {check_interval}s."
                    )
                    fetch_log_time = curr_time
                time.sleep(check_interval)
                continue

            if round_num >= self.state.round:
                get_logger().info(f"🐝 Joining round: {round_num}")
                check_backoff = check_interval
                self.state.round = round_num
                return
            else:
                get_logger().info(
                    f"Already finished round: {round_num}. Next check in {check_backoff}s."
                )
                time.sleep(check_backoff)
                check_backoff = min(check_backoff * 2, max_check_interval)

            if round_num == self.max_round - 1:
                return

        get_logger().info("Training timed out!")
