from itertools import chain
import numpy as np
import cv2 as cv
from PIL import Image, ImageDraw, ImageFont
from dataclasses import dataclass
import customtkinter
from matplotlib import cm
from datetime import datetime
import tempfile
import matplotlib.pyplot as plt
import csv
import copy

HEATMAP_WINDOW = 20


# Utilities
def get_image_from_cap(cap):
    _, cv_img = cap.read()
    cv_img = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv_img)
    return pil_img


def draw_mask_on_image(img, mask_coordinate):
    draw = ImageDraw.Draw(img)
    draw.polygon(mask_coordinate, outline="red", fill="red")


def build_mask_from_coordinate(height, width, mask_coordinate):
    mask_contour = np.array(mask_coordinate)
    mask = np.full([height, width], 255, dtype=np.uint8)
    cv.fillPoly(mask, pts=[mask_contour], color=(0))
    return mask


def get_frames(cap, start_frame, end_frame):
    cap.set(cv.CAP_PROP_POS_FRAMES, float(start_frame))
    frames = [cap.read()[1] for _ in range(end_frame - start_frame)]
    return frames


def precomp_ref(frames):
    # Build cumulative running-maximum reference per frame (groundtruth-like)
    ref_image_precomputed = []
    ref = cv.cvtColor(frames[0], cv.COLOR_BGR2GRAY)
    ref = np.array(ref)
    for f in frames[1:]:
        current = cv.cvtColor(f, cv.COLOR_BGR2GRAY)
        current_np = np.array(current)
        np.maximum(ref, current_np, out=ref)
        ref_image_precomputed.append(ref.copy())
    return ref_image_precomputed


def append_colorbar_below(image, colorbar):
    # Ensure both images have the same width
    if image.width != colorbar.width:
        colorbar = colorbar.resize((image.width, colorbar.height))

    # Create a new image with the combined height
    combined_height = image.height + colorbar.height
    combined_image = Image.new("RGB", (image.width, combined_height))

    # Paste the images into the new image
    combined_image.paste(image, (0, 0))  # Paste the main image at the top
    combined_image.paste(colorbar, (0, image.height))  # Paste the colorbar below it

    return combined_image


def add_text_to_image(
    image, text, position="bottom", text_color="black", font_size=20, bg_color=None
):
    font = (
        ImageFont.load_default()
    )  # Fallback to default font if specified font not found

    # Create a Draw object
    draw = ImageDraw.Draw(image)

    # Calculate text dimensions using textbbox
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Adjust canvas size for text
    if position in ["top", "bottom"]:
        new_height = image.height + text_height + 10
        new_image = Image.new("RGB", (image.width, new_height), bg_color or "white")
        if position == "top":
            text_y = 5
            image_y = text_height + 10
        else:
            text_y = image.height + 5
            image_y = 0
        new_image.paste(image, (0, image_y))
    elif position == "center":
        new_image = image.copy()
        text_y = (image.height - text_height) // 2
    else:
        raise ValueError("Invalid position. Choose 'top', 'bottom', or 'center'.")

    # Center the text horizontally
    text_x = (image.width - text_width) // 2

    # Draw the text on the new image
    draw = ImageDraw.Draw(new_image)
    draw.text((text_x, text_y), text, fill=text_color, font=font)

    return new_image


class VideoPlayer:
    def __init__(
        self,
        video_data,
        tkinter_frame1,
        tkinter_frame2,
        tkinter_slider,
        label_slider_frame1,
        color_bar,
    ) -> None:
        self.video_data = video_data
        self.current_frame = 0
        self.tkinter_frame1 = tkinter_frame1
        self.tkinter_frame2 = tkinter_frame2
        self.tkinter_slider = tkinter_slider
        self.tkinter_label_frame1 = label_slider_frame1
        self.cap = cv.VideoCapture(video_data.file_path)
        self.framecount = self.cap.get(cv.CAP_PROP_FRAME_COUNT)
        self.playing = True
        self.mode = "binary"  # or "heatmap"
        self.thresholding_method = "triangle"  # or "binary"
        self.binary_thresholding_param = 15
        self.current_processed_image = None
        self.mask = build_mask_from_coordinate(
            self.video_data.height,
            self.video_data.width,
            self.video_data.mask_coordinate,
        )
        self.color_bar = color_bar
        self.color_bar_default = copy.copy(color_bar)

        # Persistent heatmap state (leaky integrator)
        self.heatmap = np.zeros(
            (self.video_data.height, self.video_data.width), dtype=np.float32
        )
        self.HEATMAP_COEF_FADE_IN = 1.0
        self.HEATMAP_COEF_FADE_OUT = 0.90
        # persistence presence acknowledged if blob persists ~1 second
        self.HEATMAP_CEIL = self.HEATMAP_COEF_FADE_IN * 60.0

        # Persistent background reference (running max like groundtruth)
        self.ref = None
        self.use_persistent_ref = True

    def get_heat_map(self, start_frame, end_frame):
        cues = [
            self.calc_binary(self.cap, frame_i)[1]
            for frame_i in range(start_frame, end_frame)
        ]
        cues_sum = np.sum(cues, axis=0)  # sum

        # normalize
        min_value = np.min(cues_sum)
        max_value = np.max(cues_sum)
        normalized_array = (cues_sum - min_value) / (max_value - min_value) * 255
        normalized_array = normalized_array.astype(np.uint8)

        colormap = cm.get_cmap("nipy_spectral")  # You can use 'jet', 'plasma', etc.
        colored_array = colormap(
            normalized_array / 255.0
        )  # Normalize array to 0-1 for colormap

        # Matplotlib returns an RGBA image, convert to RGB
        colored_array_rgb = (colored_array[:, :, :3] * 255).astype(np.uint8)

        # Create the PIL image from the RGB array
        image = Image.fromarray(colored_array_rgb, "RGB")

        image = append_colorbar_below(image, self.color_bar)

        return image

    def on_the_fly_ref(self, frame_i, window_size=5):
        # Build a running per-pixel maximum over a short window (unmasked during accumulation),
        # then apply the ROI mask to the final reference like groundtruth.
        start_frame_i = max(0, frame_i - window_size)
        base = cv.cvtColor(self.video_data.frames[start_frame_i], cv.COLOR_BGR2GRAY)
        ref_image_np = np.array(base)
        for f in self.video_data.frames[start_frame_i + 1 : frame_i]:
            current = cv.cvtColor(f, cv.COLOR_BGR2GRAY)
            current_np = np.array(current)
            np.maximum(ref_image_np, current_np, out=ref_image_np)
        # Apply mask to the reference after accumulation
        ref_image_np = cv.bitwise_and(ref_image_np, self.mask)
        return ref_image_np

    def calc_binary(self, cap, frame_i, return_img=True):
        ret, current_col = cap.read()
        current = cv.cvtColor(current_col, cv.COLOR_BGR2GRAY)

        if self.use_persistent_ref:
            # Groundtruth-like running max background, masked
            if self.ref is None:
                self.ref = current.copy()
            else:
                np.maximum(self.ref, current, out=self.ref)
            ref_masked = cv.bitwise_and(self.ref, self.mask)
            cue = cv.absdiff(current, ref_masked)
        else:
            # Fallback to short-window ref (on-the-fly) or precomputed
            if self.video_data.ref_precomp:
                ref_image = self.video_data.ref_precomp[max(0, frame_i - 1)]
            else:
                ref_image = self.on_the_fly_ref(frame_i)
            # Ensure masked reference for differencing
            ref_image_masked = cv.bitwise_and(ref_image, self.mask)
            cue = cv.absdiff(current, ref_image_masked)

        if self.thresholding_method == "triangle":
            # Align maxval with binary path for consistent scaling/integration
            ret, cue = cv.threshold(cue, 0, 250, cv.THRESH_TRIANGLE)
        elif self.thresholding_method == "binary":
            ret, cue = cv.threshold(
                cue, self.binary_thresholding_param, 250, cv.THRESH_BINARY
            )

        # Apply mask after thresholding to match groundtruth flow
        cue = cv.bitwise_and(cue, self.mask)

        if return_img:
            # make it faster by skipping this if not needed
            img = Image.fromarray(cue).convert("RGB")
        else:
            img = None
        return img, cue

    def reset_heatmap(self):
        # Clear persistent heatmap state
        self.heatmap.fill(0.0)

    def reset_ref(self):
        # Clear persistent background reference
        self.ref = None

    def _update_heatmap(self, cue_bin):
        # Update leaky integrator heatmap using thresholded cue (0 or 250)
        self.heatmap += self.HEATMAP_COEF_FADE_IN * (cue_bin.astype(np.float32) / 250.0)
        self.heatmap -= self.HEATMAP_COEF_FADE_OUT
        np.maximum(self.heatmap, 0.0, out=self.heatmap)

    def _render_heatmap_image(self):
        # Prepare a clipped copy for normalization and coloring
        heatmapf = np.clip(self.heatmap, 0.0, self.HEATMAP_CEIL)
        # Anchor normalization baseline to keep display consistent
        if heatmapf.size > 0:
            heatmapf[0, 0] = 0.0

        min_value = float(np.min(heatmapf))
        max_value = float(np.max(heatmapf))
        if max_value - min_value > 1e-6:
            normalized_array = (heatmapf - min_value) / (max_value - min_value) * 255.0
        else:
            normalized_array = np.zeros_like(heatmapf)
        normalized_array = normalized_array.astype(np.uint8)

        colormap = cm.get_cmap("nipy_spectral")
        colored_array = colormap(normalized_array / 255.0)
        colored_array_rgb = (colored_array[:, :, :3] * 255).astype(np.uint8)
        image = Image.fromarray(colored_array_rgb, "RGB")

        image = append_colorbar_below(image, self.color_bar)
        return image

    def show_frame(self, frame_i, set_slider=True):
        if set_slider:
            self.tkinter_slider.set(frame_i)
        self.cap.set(cv.CAP_PROP_POS_FRAMES, float(frame_i))
        pil_img = get_image_from_cap(self.cap)

        draw_mask_on_image(pil_img, self.video_data.mask_coordinate)

        frame_image = customtkinter.CTkImage(
            light_image=pil_img,
            dark_image=pil_img,
            size=(int(640 * 0.75), int(480 * 0.75)),
        )
        self.tkinter_frame1.configure(image=frame_image)
        self.tkinter_frame1.update()

        pil_bin_img, cue_bin = self.calc_binary(self.cap, frame_i)
        # buffer cue_bin to video_data
        if self.mode == "heatmap":
            threshold = ""
            if self.thresholding_method == "binary":
                threshold = f", threshold value: {self.binary_thresholding_param}"

            self.color_bar = add_text_to_image(
                self.color_bar_default,
                (
                    f"file: {self.video_data.file_name}, "
                    f"method: {self.thresholding_method}{threshold}, "
                    f"time: {frame_i / self.video_data.fps:.3f} s"
                ),
            )
            # Update persistent heatmap and render
            self._update_heatmap(cue_bin)
            pil_bin_img = self._render_heatmap_image()

        self.current_processed_image = pil_bin_img
        bin_image = customtkinter.CTkImage(
            light_image=pil_bin_img,
            dark_image=pil_bin_img,
            size=(int(640 * 0.75), int(480 * 0.75)),
        )
        self.tkinter_frame2.configure(image=bin_image)
        self.tkinter_frame2.update()
        self.tkinter_label_frame1.configure(
            text=f"Time: {frame_i / self.video_data.fps:.3f} s (frame {frame_i})"
        )

    def set_frame_to(self, frame_i, set_slider=True):
        self.show_frame(frame_i, set_slider)
        if set_slider:
            self.current_frame = frame_i
        else:
            self.current_frame = int(self.tkinter_slider.get())

    def play(self):
        self.playing = True
        self.tkinter_slider
        start_frame = int(self.tkinter_slider.get())
        for frame_i in range(start_frame, self.video_data.end_frame):
            # time.sleep(1/30)
            # print(f"showing frame {frame_i}")
            self.show_frame(frame_i)
            self.current_frame += 1
            if self.playing == False:
                break

    def pause(self, frame_i=None):
        self.playing = False
        # time.sleep(0.5)
        # if frame_i:
        #     self.show_frame(int(frame_i))
        # else:
        #     self.show_frame(self.current_frame)

    def stop(self):
        self.playing = False
        self.show_frame(self.video_data.start_frame)
        self.current_frame = self.video_data.start_frame

    def release(self):
        self.cap.release()


@dataclass
class VideoData:
    file_path: str
    file_name: str
    width: int
    height: int
    fps: int
    frame_length: int
    ref_image: Image
    start_frame: int
    end_frame: int
    ref_precomp: list = None
    mask_coordinate: list = None
    frames: list = None


class OpenFileModel:
    def __init__(self) -> None:
        self.file_name = None
        self.file_path = None

    def set_file_path(self, file_path):
        self.file_path = file_path
        self.file_name = file_path.split("/")[-1]

    def load_video(self):
        cap = cv.VideoCapture(self.file_path)
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv.CAP_PROP_FPS)
        frame_length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        # Align with trackandcount: start from frame 0
        startframe = int(0)
        cap.set(cv.CAP_PROP_POS_FRAMES, float(startframe))
        pil_img = get_image_from_cap(cap)
        lastframe = int(frame_length)

        frames = get_frames(cap, startframe, lastframe)

        # DISABLE ref precomp TODO: CLEANING
        ref_precomp = None  # precomp_ref(frames)
        video_data = VideoData(
            file_path=self.file_path,
            file_name=self.file_name,
            width=width,
            height=height,
            fps=fps,
            frame_length=frame_length,
            ref_image=pil_img.convert("L"),
            start_frame=startframe,
            end_frame=lastframe,
            ref_precomp=ref_precomp,
            frames=frames,
        )
        return video_data


class MainWindowModel:
    def __init__(self) -> None:
        self.video_data = None
        self.video_player = None

    def set_video_data(self, video_data):
        self.video_data = video_data

    def instantiate_video_player(
        self,
        tkinter_frame1,
        tkinter_frame2,
        tkinter_slider,
        label_slider_frame1,
        color_bar,
    ):
        self.video_player = VideoPlayer(
            self.video_data,
            tkinter_frame1,
            tkinter_frame2,
            tkinter_slider,
            label_slider_frame1,
            color_bar,
        )


class MaskingModel:
    def __init__(self) -> None:
        self.mask_coordinate = []

    def add_mask_coordinate(self, coordinate):
        self.mask_coordinate.append(coordinate)
        if len(self.mask_coordinate) == 2:
            self.mask_coordinate.append(coordinate)

    def pop_mask_coordinate(self):
        if len(self.mask_coordinate) != 0:
            self.mask_coordinate.pop()

    def clear_mask_coordinate(self):
        self.mask_coordinate.clear()

    def get_mask_coordinate(self):
        return list(chain(self.mask_coordinate))


class ResultProcessModel:
    pass


class ProcessingModel:
    def calculate_count(self, video_player, start_time, end_time):
        end_time = end_time
        # Keep video FPS for mapping seconds->frame indices, but use
        # a fixed 60 fps timebase and per-second bucketing to match trackandcount
        fps_video = float(video_player.video_data.fps)
        fps_base = 60  # trackandcount timebase
        print("calculate model")
        print(f"start {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Compute absolute frame indices
        start_frame = int(float(start_time) * fps_video)
        end_frame = int((float(end_time)) * fps_video)

        count_per_second = []
        frame_log = []  # list of tuples: (time_sec, count, hmin, hmax, cmx, cmy, hmx, hmy)
        tempcount = np.zeros(fps_base, dtype=np.uint)

        # Reset persistent heatmap and background; enable persistent ref
        try:
            video_player.reset_heatmap()
            video_player.reset_ref()
        except Exception:
            pass
        video_player.use_persistent_ref = True

        # Loop over every frame from start_frame up to, but not including, end_frame
        for frame_i in range(start_frame, end_frame):
            # frame index relative to processing start; used for 60-fps bucketing
            framenum_rel = frame_i - start_frame
            idx60 = framenum_rel % fps_base
            # Seek to the correct frame
            video_player.cap.set(cv.CAP_PROP_POS_FRAMES, float(frame_i))
            _, cue = video_player.calc_binary(
                video_player.cap, frame_i, return_img=False
            )
            # Handle OpenCV version differences in findContours
            try:
                contours, hierarchy = cv.findContours(
                    cue, cv.RETR_LIST, cv.CHAIN_APPROX_NONE
                )
            except ValueError:
                _, contours, hierarchy = cv.findContours(
                    cue, cv.RETR_LIST, cv.CHAIN_APPROX_NONE
                )

            # Count contours for this frame
            tempcount[idx60] = len(contours)

            # Current presence centroid from binary cue
            cmoments = cv.moments(cue, binaryImage=True)
            mass = cmoments.get("m00", 0.0) or 0.0
            if mass == 0:
                mass = 1.0
            cmx = int(cmoments.get("m10", 0.0) / mass)
            cmy = int(cmoments.get("m01", 0.0) / mass)

            # Update leaky heatmap using the same cue
            try:
                video_player._update_heatmap(cue)
            except Exception:
                pass

            # Heatmap stats and centroid
            hmin = float(np.min(video_player.heatmap)) if hasattr(video_player, "heatmap") else 0.0
            hmax = float(np.max(video_player.heatmap)) if hasattr(video_player, "heatmap") else 0.0
            try:
                hmoments = cv.moments(video_player.heatmap)
                hmass = hmoments.get("m00", 0.0) or 0.0
            except Exception:
                hmoments = {"m00": 0.0, "m10": 0.0, "m01": 0.0}
                hmass = 0.0
            if hmass == 0:
                hmass = 1.0
            hmx = int(hmoments.get("m10", 0.0) / hmass)
            hmy = int(hmoments.get("m01", 0.0) / hmass)

            # Log per-frame metrics to match groundtruth intent
            # Match trackandcount timebase: seconds = frames / 60
            time_sec = frame_i / fps_base
            frame_log.append((
                round(time_sec, 3),
                int(len(contours)),
                hmin,
                hmax,
                int(cmx),
                int(cmy),
                int(hmx),
                int(hmy),
            ))

            # Once we have collected one second's worth of frames, average and reset
            if idx60 == fps_base - 1:
                count_per_second.append(tempcount.mean())
                tempcount.fill(0)

        print(f"end {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        video_player.use_persistent_ref = False

        # Build metadata string
        meta_data = f"file: {video_player.video_data.file_name}"
        meta_data += f"\nthresholding method: {video_player.thresholding_method}"
        if video_player.thresholding_method == "binary":
            meta_data += f", threshold value: {video_player.binary_thresholding_param}"

        return count_per_second, meta_data, frame_log


class ResultModel:
    def _create_plot(self, count_per_second, meta_data):
        """Create and return a matplotlib Figure for the given data and metadata."""
        x = np.arange(1, len(count_per_second) + 1)
        fig, ax = plt.subplots()
        ax.plot(x, count_per_second)
        ax.set_xlabel("time (second)")
        ax.set_ylabel("count")
        fig.subplots_adjust(bottom=0.2)
        fig.text(0.03, 0.03, meta_data, ha="left", fontsize=10)

        n = len(count_per_second)
        if 15 < n <= 30:
            step = 5
        elif n > 30:
            step = 15
        else:
            step = 1

        ax.set_xticks(np.arange(1, n + step, step))
        return fig

    def _format_metadata_lines(self, meta_data):
        """Return a list of metadata lines, each prefixed with '#' for CSV."""
        return [f"#{line}" for line in meta_data.split("\n")]

    def generate_plot_image(self, count_per_second, meta_data):
        fig = self._create_plot(count_per_second, meta_data)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig.savefig(tmp.name)
        plt.close(fig)

        pil_img = Image.open(tmp.name)
        tk_img = customtkinter.CTkImage(
            light_image=pil_img,
            dark_image=pil_img,
            size=(640, 480)
        )
        tmp.close()
        return tk_img

    def save_image(self, file_path, count_per_second, meta_data):
        ext = file_path.rsplit(".", 1)[-1]
        fig = self._create_plot(count_per_second, meta_data)
        fig.savefig(file_path, format=ext, dpi=300)
        plt.close(fig)

    def save_csv(self, file_path, count_per_second, meta_data, frame_log):
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            # Metadata as commented lines
            for line in self._format_metadata_lines(meta_data):
                writer.writerow([line])

            # Groundtruth-like header
            writer.writerow([
                "time",
                "count",
                "heatmap_min",
                "heatmap_max",
                "detection_cx",
                "detection_cy",
                "heatmap_cx",
                "heatmap_cy",
            ])

            # Per-frame logs
            for row in frame_log:
                writer.writerow(row)

            # Per-second summary (optional, like groundtruth averages)
            writer.writerow([])
            writer.writerow(["# per-second summary", "sec", "avg_count"])
            for i, avg in enumerate(count_per_second, start=1):
                writer.writerow([i, float(avg)])
