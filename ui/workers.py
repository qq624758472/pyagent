"""后台线程：Agent 执行与模型列表拉取

UI 线程只碰界面：推理/网络一律在子线程，通过 pyqtSignal 回主线程。
"""
import threading

from PyQt5.QtCore import QThread, pyqtSignal

from core.config import Config
from core.llm_client import LLMClient, LLMError
from core.log import get_logger

log = get_logger("ui.workers")


class AgentWorker(QThread):
    """Agent 执行线程：把事件流通过信号送回主线程（UI 只碰主线程）"""
    agent_event = pyqtSignal(dict)

    def __init__(self, agent, text, confirm=None, parent=None):
        super().__init__(parent)
        self.agent = agent
        self.text = text
        self.confirm = confirm
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        try:
            for ev in self.agent.run(
                    self.text,
                    should_stop=lambda: self._stop.is_set(),
                    progress=lambda msg: self.agent_event.emit(
                        {"type": "progress", "content": msg}),
                    confirm=self.confirm):
                self.agent_event.emit(ev)
        except Exception as e:                    # 线程内异常不能静默丢失
            log.exception("AgentWorker 线程未捕获异常")
            self.agent_event.emit({"type": "error", "content": f"内部错误: {e}"})
        finally:
            self.agent_event.emit({"type": "done"})


class FetchModelsWorker(QThread):
    """拉取模型列表（网络请求不能阻塞 UI 线程）"""
    ok = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, api_key, base_url, parent=None):
        super().__init__(parent)
        self.api_key = api_key
        self.base_url = base_url

    def run(self):
        try:
            cfg = Config()
            cfg.api_key = self.api_key
            cfg.base_url = self.base_url
            ids = LLMClient(cfg).list_models()
            self.ok.emit(ids)
        except LLMError as e:
            self.failed.emit(str(e))
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
