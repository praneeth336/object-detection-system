import os
import sys
import tempfile
import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.detector import YOLODetector
from core.roi import ROIManager, LineCounter
from core.utils import load_config, export_detections_csv, export_detections_json
from core.trainer import YOLOTrainer

# Streamlit Page Config
st.set_page_config(
    page_title="YOLO Object Detection & Tracking System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_detector(model_name: str, conf: float, iou: float):
    return YOLODetector(model_path=model_name, conf_threshold=conf, iou_threshold=iou)

def main():
    st.markdown('<div class="main-header">🎯 YOLO Object Detection & Tracking System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time object detection, multi-object tracking, region of interest analytics, and custom model training.</div>', unsafe_allow_html=True)

    config = load_config()
    available_models = [m["name"] for m in config.get("models", {}).get("available", [{"name": "yolov8n.pt"}])]

    # Sidebar Controls
    st.sidebar.header("⚙️ Model & Detection Settings")
    
    selected_model = st.sidebar.selectbox("Select Model Architecture", available_models, index=0)
    conf_thresh = st.sidebar.slider("Confidence Threshold", min_value=0.05, max_value=1.0, value=0.25, step=0.05)
    iou_thresh = st.sidebar.slider("IoU (NMS) Threshold", min_value=0.05, max_value=1.0, value=0.45, step=0.05)

    # Instantiate Detector
    try:
        detector = get_detector(selected_model, conf_thresh, iou_thresh)
        detector.update_parameters(conf_threshold=conf_thresh, iou_threshold=iou_thresh, model_path=selected_model)
    except Exception as e:
        st.sidebar.error(f"Error loading model {selected_model}: {e}")
        return

    # Class Filter
    class_names = list(detector.class_names.values()) if detector.class_names else []
    selected_classes_names = st.sidebar.multiselect("Filter Target Classes (Empty = All)", options=class_names, default=[])
    
    target_class_ids = None
    if selected_classes_names:
        name_to_id = {v: k for k, v in detector.class_names.items()}
        target_class_ids = [name_to_id[name] for name in selected_classes_names if name in name_to_id]

    # Analytics / ROI Toggles
    st.sidebar.header("📐 ROI & Tracking Analytics")
    enable_tracking = st.sidebar.checkbox("Enable Multi-Object Tracking (MOT)", value=True)
    enable_roi = st.sidebar.checkbox("Enable Region of Interest (ROI Zone)", value=False)
    enable_line_crossing = st.sidebar.checkbox("Enable Line Crossing Counter", value=False)

    roi_manager = None
    if enable_roi:
        st.sidebar.markdown("**ROI Polygon Points (Normalized 0.0 - 1.0)**")
        roi_pts_str = st.sidebar.text_area("Points (x,y per line)", value="0.2, 0.2\n0.8, 0.2\n0.8, 0.8\n0.2, 0.8")
        try:
            pts = []
            for line in roi_pts_str.strip().split("\n"):
                if line.strip():
                    x, y = map(float, line.split(","))
                    pts.append([x, y])
            roi_manager = ROIManager(pts)
        except Exception:
            st.sidebar.warning("Invalid ROI points format. Use 'x, y' per line.")

    line_counter = None
    if enable_line_crossing:
        line_counter = LineCounter(line_pt1=[0.1, 0.5], line_pt2=[0.9, 0.5])

    # Main Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Image Detection", "🎥 Video & Stream Processing", "🏋️ Custom Training", "📊 Analytics & Export"])

    # TAB 1: Image Detection
    with tab1:
        st.subheader("Image Object Detection")
        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png", "webp"])
        
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            start_t = time.time()
            result = detector.detect_frame(
                frame=image,
                enable_tracking=enable_tracking,
                target_classes=target_class_ids,
                roi_manager=roi_manager,
                line_counter=line_counter,
                annotate=True
            )
            latency_ms = (time.time() - start_t) * 1000

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Input**")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

            with col2:
                st.markdown("**YOLO Annotated Detection**")
                if result.annotated_image is not None:
                    st.image(cv2.cvtColor(result.annotated_image, cv2.COLOR_BGR2RGB), use_container_width=True)

            # Metrics display
            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("Total Objects Detected", result.total_count)
            mcol2.metric("Inference Latency", f"{latency_ms:.1f} ms")
            mcol3.metric("Detected Classes", len(result.counts_by_class))

            if result.counts_by_class:
                st.markdown("##### Detected Object Counts")
                counts_df = pd.DataFrame(list(result.counts_by_class.items()), columns=["Class", "Count"])
                st.dataframe(counts_df, use_container_width=True)

            if result.detections:
                dets_df = pd.DataFrame(result.detections)
                st.markdown("##### Detailed Bounding Box Data")
                st.dataframe(dets_df, use_container_width=True)

                # Export downloads
                csv_path = export_detections_csv(result.detections, "outputs/latest_image_detections.csv")
                with open(csv_path, "rb") as f:
                    st.download_button("📥 Download Detections CSV", f, file_name="image_detections.csv", mime="text/csv")

    # TAB 2: Video & Stream Processing
    with tab2:
        st.subheader("Video File & Stream Object Detection")
        source_type = st.radio("Select Video Input Source", ["Video File Upload", "Webcam Feed (Local)"], horizontal=True)

        if source_type == "Video File Upload":
            video_file = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov", "mkv"])
            if video_file is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(video_file.read())
                tfile.close()

                st_video_placeholder = st.empty()
                st_info_placeholder = st.empty()
                st_chart_placeholder = st.empty()

                if st.button("▶️ Start Video Detection"):
                    cap = cv2.VideoCapture(tfile.name)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps_val = cap.get(cv2.CAP_PROP_FPS) or 30

                    progress_bar = st.progress(0)
                    all_video_detections = []
                    counts_over_time = []
                    frame_idx = 0

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        res = detector.detect_frame(
                            frame=frame,
                            frame_index=frame_idx,
                            enable_tracking=enable_tracking,
                            target_classes=target_class_ids,
                            roi_manager=roi_manager,
                            line_counter=line_counter,
                            annotate=True
                        )

                        all_video_detections.extend(res.detections)
                        counts_over_time.append({"frame": frame_idx, "count": res.total_count, "fps": res.fps})

                        if res.annotated_image is not None:
                            rgb_frame = cv2.cvtColor(res.annotated_image, cv2.COLOR_BGR2RGB)
                            st_video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                        st_info_placeholder.text(f"Processing Frame {frame_idx}/{frame_count} | Objects: {res.total_count} | FPS: {res.fps}")
                        
                        if frame_count > 0:
                            progress_bar.progress(min(1.0, frame_idx / frame_count))

                        frame_idx += 1

                    cap.release()
                    st.success("Video processing complete!")

                    if counts_over_time:
                        counts_df = pd.DataFrame(counts_over_time)
                        st_chart_placeholder.line_chart(counts_df.set_index("frame")[["count", "fps"]])

        elif source_type == "Webcam Feed (Local)":
            st.info("Ensure a local webcam device (Index 0) is connected.")
            run_cam = st.checkbox("Turn On Webcam")
            cam_placeholder = st.empty()
            cam_info = st.empty()

            if run_cam:
                cap = cv2.VideoCapture(0)
                frame_idx = 0
                while run_cam and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to capture image from webcam.")
                        break

                    res = detector.detect_frame(
                        frame=frame,
                        frame_index=frame_idx,
                        enable_tracking=enable_tracking,
                        target_classes=target_class_ids,
                        roi_manager=roi_manager,
                        line_counter=line_counter,
                        annotate=True
                    )

                    if res.annotated_image is not None:
                        rgb_frame = cv2.cvtColor(res.annotated_image, cv2.COLOR_BGR2RGB)
                        cam_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                    cam_info.markdown(f"**Live Feed FPS:** {res.fps} | **Current Object Count:** {res.total_count}")
                    frame_idx += 1
                cap.release()

    # TAB 3: Custom Model Training
    with tab3:
        st.subheader("Fine-Tune YOLO on Custom Dataset")
        st.markdown("Train a custom YOLO object detector by supplying a `data.yaml` file pointing to annotated images and labels.")

        data_yaml_path = st.text_input("Path to dataset YAML file (data.yaml)", value="sample_data/data.yaml")
        epochs = st.number_input("Number of Epochs", min_value=1, max_value=500, value=25)
        batch_size = st.number_input("Batch Size", min_value=1, max_value=64, value=16)
        img_size = st.select_slider("Image Size (pixels)", options=[320, 416, 512, 640, 1024], value=640)

        if st.button("🚀 Start Training Pipeline"):
            if not os.path.exists(data_yaml_path):
                st.error(f"Dataset YAML file not found at: {data_yaml_path}")
            else:
                with st.spinner("Training YOLO model in progress..."):
                    try:
                        trainer = YOLOTrainer(base_model=selected_model)
                        res = trainer.train(
                            data_yaml=data_yaml_path,
                            epochs=epochs,
                            imgsz=img_size,
                            batch_size=batch_size
                        )
                        st.success(f"Training completed! Weights saved to: {res.get('best_weights')}")
                        st.json(res)
                    except Exception as ex:
                        st.error(f"Training failed: {ex}")

    # TAB 4: Analytics & Export
    with tab4:
        st.subheader("System Analytics & Configuration")
        st.json(config)

if __name__ == "__main__":
    main()
