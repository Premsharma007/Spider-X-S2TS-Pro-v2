import threading
import time
import psutil
import GPUtil
from typing import Dict, Any

class ResourceMonitor:
    def __init__(self):
        self.metrics = {
            "cpu": 0,
            "memory": 0,
            "gpu": 0,
            "gpu_memory": 0,
            "disk": 0,
            "cpu_html": "",
            "gpu_html": ""
        }
        self.is_monitoring = False
        self.thread = None
    
    def start_monitoring(self, interval: int = 5):
        """Start monitoring system resources in a background thread."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop, args=(interval,), daemon=True)
        self.thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring system resources."""
        self.is_monitoring = False
        if self.thread:
            self.thread.join(timeout=2.0)
    
    def _monitor_loop(self, interval: int):
        """Monitor system resources in a loop."""
        while self.is_monitoring:
            self.update_metrics()
            time.sleep(interval)
    
    def update_metrics(self):
        """Update all resource metrics and build HTML for CPU/GPU bars."""
        # CPU usage
        self.metrics["cpu"] = psutil.cpu_percent(interval=0.1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.metrics["memory"] = memory.percent
        
        # GPU usage (if available)
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                self.metrics["gpu"] = gpus[0].load * 100
                self.metrics["gpu_memory"] = gpus[0].memoryUtil * 100
            else:
                self.metrics["gpu"] = 0
                self.metrics["gpu_memory"] = 0
        except:
            self.metrics["gpu"] = 0
            self.metrics["gpu_memory"] = 0
        
        # Disk usage (of current drive)
        disk = psutil.disk_usage('/')
        self.metrics["disk"] = disk.percent

        # Build HTML bars
        def neon_bar(label, value, color):
            return f"""
            <div style='margin-bottom:6px;'>
                <span style='color:#94a3b8;font-size:0.9em;'>{label}: {value:.1f}%</span>
                <div style='background:#1e293b;border-radius:6px;height:16px;overflow:hidden;'>
                    <div style='width:{value}%;height:100%;background:{color};box-shadow:0 0 8px {color};transition:width 0.3s ease;'></div>
                </div>
            </div>
            """

        self.metrics["cpu_html"] = neon_bar("CPU Usage", self.metrics["cpu"], "#38bdf8")
        self.metrics["gpu_html"] = neon_bar("GPU Usage", self.metrics["gpu"], "#f43f5e")
        self.metrics["memory_html"] = neon_bar("Memory Usage", self.metrics["memory"], "#1083b9")
        self.metrics["disk_html"] = neon_bar("Disk Usage", self.metrics["disk"], "#0b97f5")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current resource metrics, including prebuilt HTML for CPU/GPU."""
        return self.metrics.copy()
