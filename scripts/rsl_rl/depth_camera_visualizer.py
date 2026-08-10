"""Small depth camera visualizer for yaw_bot."""

from __future__ import annotations

import os
from pathlib import Path


class DepthCameraVisualizer:
    """Display the first environment's raw depth camera output."""

    def __init__(
        self,
        env,
        *,
        enabled: bool,
        interval: int,
        env_index: int = 0,
        window_name: str = "YawBot depth camera",
        output_dir: str | None = None,
    ):
        self.env = env
        self.enabled = enabled
        self.interval = max(1, int(interval))
        self.env_index = env_index
        self.window_name = window_name
        self.step_count = 0
        self._cv2 = None
        self._plt = None
        self._figure = None
        self._axes = None
        self._image_artist = None
        self._warned = False
        self._window_ready = False
        self._output_dir = Path(output_dir) if output_dir else None

        if self._output_dir is not None:
            self._output_dir.mkdir(parents=True, exist_ok=True)

        if self.enabled:
            self._initialize_backend()

    def update(self, *, force: bool = False) -> None:
        """Refresh the preview window when the interval is reached."""
        if not self.enabled:
            return

        self.step_count += 1
        if not force and self.step_count % self.interval != 0:
            return

        try:
            depth = self.env.depth_camera.data.output["distance_to_image_plane"][self.env_index]
            max_distance = float(self.env.cfg.depth_max_distance)
            image = self._depth_to_normalized_numpy(depth, max_distance)

            if self._cv2 is not None:
                self._update_opencv(image)
            elif self._plt is not None:
                self._update_matplotlib(image)

            if self._output_dir is not None:
                self._save_latest(image)
        except Exception as exc:
            if not self._warned:
                print(f"[WARN] Depth camera visualization update failed: {exc}", flush=True)
                self._warned = True

    def close(self) -> None:
        if self._cv2 is not None and self._window_ready:
            self._cv2.destroyWindow(self.window_name)
        if self._plt is not None and self._figure is not None:
            self._plt.close(self._figure)

    def _initialize_backend(self) -> None:
        print(
            f"[INFO] Depth camera visualization enabled: interval={self.interval}, env_index={self.env_index}",
            flush=True,
        )
        try:
            import cv2  # noqa: PLC0415

            gui_line = next((line for line in cv2.getBuildInformation().splitlines() if "GUI:" in line), "")
            if "NONE" not in gui_line:
                self._cv2 = cv2
                print("[INFO] Depth camera preview backend: OpenCV.", flush=True)
                return
            print("[INFO] OpenCV has no GUI backend; falling back to Matplotlib.", flush=True)
        except Exception as exc:
            print(f"[INFO] OpenCV preview unavailable; falling back to Matplotlib: {exc}", flush=True)

        try:
            os.environ.setdefault("MPLCONFIGDIR", "/tmp/yawbot-mpl")
            import matplotlib  # noqa: PLC0415

            matplotlib.use("TkAgg", force=True)
            import matplotlib.pyplot as plt  # noqa: PLC0415

            self._plt = plt
            plt.ion()
            print("[INFO] Depth camera preview backend: Matplotlib TkAgg.", flush=True)
        except Exception as exc:
            self._plt = None
            print(
                "[WARN] No window-capable depth preview backend is available. "
                f"Latest PNG will still be written if output_dir is set: {exc}",
                flush=True,
            )

    def _update_opencv(self, image) -> None:
        if not self._window_ready:
            self._cv2.namedWindow(self.window_name, self._cv2.WINDOW_NORMAL)
            self._cv2.resizeWindow(self.window_name, 768, 432)
            self._window_ready = True

        image_u8 = (image * 255.0).astype("uint8")
        image_color = self._cv2.applyColorMap(image_u8, self._cv2.COLORMAP_TURBO)
        self._cv2.imshow(self.window_name, image_color)
        self._cv2.waitKey(1)

    def _update_matplotlib(self, image) -> None:
        if self._figure is None:
            self._figure, self._axes = self._plt.subplots(num=self.window_name)
            self._figure.canvas.manager.set_window_title(self.window_name)
            self._image_artist = self._axes.imshow(image, cmap="turbo", vmin=0.0, vmax=1.0)
            self._axes.set_axis_off()
            self._figure.tight_layout(pad=0)
            self._window_ready = True
        else:
            self._image_artist.set_data(image)
        self._figure.canvas.draw_idle()
        self._plt.pause(0.001)

    def _save_latest(self, image) -> None:
        path = self._output_dir / "depth_camera_latest.png"
        if self._cv2 is not None:
            image_u8 = (image * 255.0).astype("uint8")
            image_color = self._cv2.applyColorMap(image_u8, self._cv2.COLORMAP_TURBO)
            self._cv2.imwrite(os.fspath(path), image_color)
        elif self._plt is not None:
            self._plt.imsave(path, image, cmap="turbo", vmin=0.0, vmax=1.0)

    def _depth_to_normalized_numpy(self, depth, max_distance: float):
        import torch  # noqa: PLC0415

        depth = torch.nan_to_num(depth.detach(), nan=max_distance, posinf=max_distance, neginf=0.0)
        depth = torch.clamp(depth, min=0.0, max=max_distance)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]

        return (1.0 - depth / max_distance).cpu().numpy()


def add_depth_camera_visualization_args(parser) -> None:
    parser.add_argument(
        "--visualize_depth_camera",
        action="store_true",
        default=False,
        help="Show the yaw_bot depth camera image in a preview window.",
    )
    parser.add_argument(
        "--depth_visualization_interval",
        type=int,
        default=10,
        help="Refresh interval, in environment steps, for --visualize_depth_camera.",
    )
