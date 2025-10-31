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
    """GameManager with 3-hour gated submissions, preserving accuracy logic."""

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
        # 修复：使用setattr安全地设置属性
        setattr(self.state, 'peer_id', self.peer_id)
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
        round_result, _ = self.coordinator.get_round_and_stage()
        # 修复：正确处理RPCResponse
        self.state.round = int(str(round_result)) if round_result is not None else 0
        self.communication.step_ = self.state.round  # initialize communication module to contract's round

        # Submission control and state trackers
        self.submit_frequency = submit_frequency  # in hours
        self.batched_signals = 0.0
        self.submitted_this_round = False
        self.cached_zero_reward_rounds = []  # zero reward cache
        self.pending_rewards = []  # aggregate per-round rewards awaiting submission
        self.sim_zero_cache = list(self.cached_zero_reward_rounds)
        # Gate submissions to every submit_frequency hours (default 3h)
        self.time_since_submit = time.time()

        # PRG Game
        self.prg_module = PRGModule(log_dir, **kwargs)
        self.prg_game = self.prg_module.prg_game

        # Disable HF push functionality
        self.hf_token = None
        self.hf_push_frequency = hf_push_frequency
        get_logger().info("Hugging Face Hub functionality is disabled")

        get_logger().info(
            f"🐱 Hello 🐱 [{get_name_from_peer_id(self.peer_id)}] 🦮 [{self.peer_id}]!"
        )
        get_logger().info(f"bootnodes: {kwargs.get('bootnodes', [])}")
        # Disabled model info logging to prevent HF-related errors
        # get_logger().info(f"Using Model: {self.trainer.model.config.name_or_path}")

        with open(os.path.join(log_dir, f"system_info.txt"), "w") as f:
            f.write(get_system_info())

    def _get_total_rewards_by_agent(self):
        rewards_by_agent = defaultdict(int)
        for stage in range(self.state.stage):
            # 修复：通过父类属性访问rewards
            rewards = super().__getattribute__('rewards')[stage]
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

    # ----------------- 新逻辑接口（严格每3小时提交一次） -----------------
    def _hook_after_rewards_updated(self):
        """奖励更新后累计信号，超过 submit_frequency 小时才触发一次提交。"""
        try:  # 添加异常处理
            rewards_by_agent = self._get_total_rewards_by_agent()
            if not rewards_by_agent:
                get_logger().warning(f"No rewards data for round {self.state.round}")
                return

            # 计算本轮按官方逻辑应提交的奖励，并累积到待提交池
            my_rewards_total = rewards_by_agent.get(self.peer_id, 0)
            if my_rewards_total > 0:
                per_round_reward = my_rewards_total + 1 + len(self.sim_zero_cache)
                self.sim_zero_cache = []
            else:
                per_round_reward = len(self.sim_zero_cache)
                if self.state.round not in self.sim_zero_cache:
                    self.sim_zero_cache.append(self.state.round)
                    if len(self.sim_zero_cache) > 3:
                        self.sim_zero_cache.pop(0)

            per_round_reward = int(max(per_round_reward, 0))
            self.pending_rewards.append(per_round_reward)
            self.batched_signals += per_round_reward

            # 时间门控：严格每 submit_frequency 小时一次
            elapsed_time_hours = (time.time() - self.time_since_submit) / 3600
            if elapsed_time_hours < self.submit_frequency:
                return

            if not self.pending_rewards:
                return

            aggregated_reward = sum(self.pending_rewards)

            # 聚合后的奖励已包含 zero cache 贡献，因此在提交前暂不叠加缓存
            self.cached_zero_reward_rounds = []

            submission_success = self._submit_current_rewards(
                aggregated_reward,
                rewards_by_agent,
            )

            if submission_success:
                # 成功后更新时间基准，保证节奏为每3小时一次
                self.time_since_submit = time.time()
                self.pending_rewards = []
                self.batched_signals = 0.0
            else:
                # 提交失败保持模拟缓存状态，便于下次重试
                self.cached_zero_reward_rounds = list(self.sim_zero_cache)
        except Exception as e:  # 添加通用异常处理
            get_logger().debug(f"Error in _hook_after_rewards_updated: {e}")

    def _submit_current_rewards(self, my_rewards, rewards_by_agent):
        """提交奖励和胜者，包含缓存 zero reward；提交均带重试与健康检查。"""
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
                    get_logger().warning(
                        f"Submit reward attempt {attempt+1} failed: {e}"
                    )
                    if attempt == 2:
                        get_logger().error(
                            f"Failed to submit reward after 3 attempts: {e}"
                        )
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
                    get_logger().warning(
                        f"Submit winners attempt {attempt+1} failed: {e}"
                    )
                    if attempt == 2:
                        get_logger().error(
                            f"Failed to submit winners after 3 attempts: {e}"
                        )
                        return False

            # zero reward 缓存由聚合逻辑维护，这里同步模拟状态
            self.cached_zero_reward_rounds = list(self.sim_zero_cache)

            self.submitted_this_round = True

            self._call_health_check()
            return True

        except Exception as e:
            get_logger().error(f"Error in _submit_current_rewards: {e}")
            return False

    def _hook_after_round_advanced(self):
        """轮次推进后，保存 HF 并阻塞等待下一轮（不做补提交）。"""
        try:  # 添加异常处理
            if self.prg_game:
                prg_history_dict = self.prg_module.prg_history_dict
                # 启用 play_prg_game_logits 以支持 judge 功能
                results_dict = self.trainer.play_prg_game_logits(prg_history_dict)
                self.prg_module.play_prg_game(results_dict, self.peer_id)
        except Exception as e:  # 添加异常处理
            get_logger().info(f"Error playing PRG game, continuing with the next round")

        self._save_to_hf()
        
        # Try to submit to chain again if necessary, but don't update our signal twice
        if not self.submitted_this_round:
            try:  # 添加异常处理
                signal_by_agent = self._get_total_rewards_by_agent()
                if not signal_by_agent:
                    get_logger().warning(f"No rewards data for round {self.state.round}")
            except Exception as e:  # 添加异常处理
                get_logger().debug(f"Error getting total rewards by agent: {e}")
                signal_by_agent = {}
            self._try_submit_to_chain(signal_by_agent)

        self.agent_block()
        self.submitted_this_round = False

    def _hook_after_game(self):
        self._save_to_hf()

    def _save_to_hf(self):
        # Hugging Face functionality is disabled
        pass

    def agent_block(
        self, check_interval=5.0, log_timeout=10.0, max_check_interval=60.0 * 5
    ):
        start_time = time.monotonic()
        fetch_log_time = start_time
        check_backoff = check_interval
        while time.monotonic() - start_time < self.train_timeout:
            curr_time = time.monotonic()
            # 按照原始实现直接访问dht属性
            _ = self.communication.dht.get_visible_maddrs(latest=True)

            try:
                round_result, stage = self.coordinator.get_round_and_stage()
                # 修复：正确处理RPCResponse
                round_num = int(str(round_result)) if round_result is not None else self.state.round
            except Exception as e:
                if curr_time - fetch_log_time > log_timeout:
                    get_logger().debug(
                        f"Could not fetch round and stage: {e}. Next check in {check_interval}s."
                    )
                    fetch_log_time = curr_time
                time.sleep(check_interval)
                continue

            # 修复：正确比较round_num和self.state.round
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

    def _call_health_check(self):
        """异步调用本地健康检查接口，不阻塞主流程"""
        def _request():
            try:
                # 使用POST方法调用健康检查API
                resp = requests.post("http://localhost:3000/api/health-check", timeout=15)
                if resp.status_code == 200:
                    get_logger().info("Health check successful")
                else:
                    get_logger().warning(
                        f"Health check returned status {resp.status_code}"
                    )
            except Exception as e:
                get_logger().warning(f"Health check failed: {e}")

        # 异步线程执行
        threading.Thread(target=_request, daemon=True).start()
