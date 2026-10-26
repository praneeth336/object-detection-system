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
from evaluate import evaluate_model

# Streamlit Page Config
st.set_page_config(
    page_title="YOLO Object Detection, Color & Distance Analytics",
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
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    .alert-banner {
        background-color: #ffebee;
        color: #c62828;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        font-weight: 600;
        margin-bottom: 1rem;
        border-left: 5px solid #d32f2f;
    }
    .safe-banner {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        font-weight: 600;
        margin-bottom: 1rem;
        border-left: 5px solid #388e3c;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_detector(model_name: str, conf: float, iou: float):
    return YOLODetector(model_path=model_name, conf_threshold=conf, iou_threshold=iou)

def main():
    st.markdown('<div class="main-header">🎯 YOLO Object Detection, Color & Spatial Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time object detection, dominant color recognition, inter-object distance estimation, tracking, and ROI analytics.</div>', unsafe_allow_html=True)

    config = load_config()
    available_models = [m["name"] for m in config.get("models", {}).get("available", [{"name": "yolov8n.pt"}])]

    # 1. Sidebar - Model Settings
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

    # 2. Sidebar - Color & Distance Features
    st.sidebar.header("🎨 Color & 📏 Distance Analytics")
    enable_color = st.sidebar.checkbox("Detect Object Dominant Color", value=True, help="Identifies dominant object color (Red, Blue, Green, etc.) using computer vision color clustering.")
    enable_distance = st.sidebar.checkbox("Measure Distance Between Objects", value=True, help="Calculates Euclidean spatial distance between detected object centroids and draws connecting measurement lines.")
    
    dist_threshold = 150
    dist_mode = "all"
    if enable_distance:
        dist_threshold = st.sidebar.slider("Proximity Alert Threshold (pixels)", min_value=30, max_value=500, value=150, step=10, help="Objects closer than this distance trigger a proximity alert with a red indicator.")
        mode_option = st.sidebar.selectbox(
            "Distance Line Connect Mode",
            ["All Object Pairs", "Proximity Alerts Only (< Threshold)", "Nearest Neighbor Only"],
            index=0
        )
        if mode_option == "Proximity Alerts Only (< Threshold)":
            dist_mode = "close_only"
        elif mode_option == "Nearest Neighbor Only":
            dist_mode = "nearest_only"
        else:
            dist_mode = "all"

    # 3. Sidebar - ROI & Tracking
    st.sidebar.header("📐 Zone & Tracking Analytics")
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🖼️ Image Detection",
        "🎥 Video & Stream Processing",
        "🎯 Model Accuracy Benchmark",
        "🏋️ Custom Training",
        "📊 Analytics & Export"
    ])

    # TAB 1: Image Detection
    with tab1:
        st.subheader("Image Object Detection, Color Recognition & Distance Estimation")
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
                detect_color=enable_color,
                measure_distance=enable_distance,
                distance_threshold=float(dist_threshold),
                distance_connect_mode=dist_mode,
                annotate=True
            )
            latency_ms = (time.time() - start_t) * 1000

            # Side by side visual display
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Input**")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

            with col2:
                st.markdown("**Annotated Detection (with Color & Distance Lines)**")
                if result.annotated_image is not None:
                    st.image(cv2.cvtColor(result.annotated_image, cv2.COLOR_BGR2RGB), use_container_width=True)

            # Metrics row
            close_pairs_count = sum(1 for p in result.distances if p.get("is_close", False))
            
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("Total Objects Detected", result.total_count)
            mcol2.metric("Inference Latency", f"{latency_ms:.1f} ms")
            
            if enable_distance and len(result.detections) > 1:
                closest_txt = f"{result.closest_distance} px" if result.closest_distance is not None else "N/A"
                mcol3.metric("Closest Object Distance", closest_txt)
                mcol4.metric("🚨 Proximity Alerts (< threshold)", close_pairs_count)
            else:
                mcol3.metric("Detected Classes", len(result.counts_by_class))
                mcol4.metric("Proximity Analysis", "Needs ≥ 2 objects" if len(result.detections) < 2 else "Disabled")

            # Proximity Alert Banner
            if enable_distance and close_pairs_count > 0:
                st.markdown(f'<div class="alert-banner">⚠️ PROXIMITY ALERT: {close_pairs_count} pair(s) of objects are within {dist_threshold} pixels of each other!</div>', unsafe_allow_html=True)
            elif enable_distance and len(result.detections) >= 2:
                st.markdown(f'<div class="safe-banner">✅ Safe Spatial Clearance: All detected objects maintain distance &ge; {dist_threshold} pixels.</div>', unsafe_allow_html=True)

            # Expandable Details: Objects & Colors
            st.markdown("---")
            det_tab, dist_tab = st.tabs(["📋 Detected Objects & Color Classification", "📏 Inter-Object Pairwise Distances"])
            
            with det_tab:
                if result.detections:
                    table_rows = []
                    for idx, d in enumerate(result.detections):
                        table_rows.append({
                            "Index": idx + 1,
                            "Track ID": d.get("track_id", "-"),
                            "Class": d.get("class_name", ""),
                            "Dominant Color": d.get("color_name", "N/A"),
                            "Color Hex": d.get("color_hex", ""),
                            "Confidence": f"{d.get('confidence', 0):.2%}",
                            "Bounding Box [x1, y1, x2, y2]": str(d.get("bbox", []))
                        })
                    df_objs = pd.DataFrame(table_rows)
                    st.dataframe(df_objs, use_container_width=True)

                    # Export downloads
                    csv_path = export_detections_csv(result.detections, "outputs/latest_image_detections.csv")
                    with open(csv_path, "rb") as f:
                        st.download_button("📥 Download Detections CSV (with Color)", f, file_name="detections_with_color.csv", mime="text/csv")
                else:
                    st.info("No objects detected in the uploaded image.")

            with dist_tab:
                if enable_distance and result.distances:
                    dist_rows = []
                    for p in result.distances:
                        dist_rows.append({
                            "Object 1": p["obj_a_label"],
                            "Object 2": p["obj_b_label"],
                            "Distance (Pixels)": f"{p['distance_px']:.1f} px",
                            "Safety Status": p["status"],
                            "Alert (< Threshold)": "🚨 YES" if p["is_close"] else "🟢 NO"
                        })
                    df_dist = pd.DataFrame(dist_rows)
                    st.dataframe(df_dist, use_container_width=True)
                elif enable_distance:
                    st.info("At least two objects are required to compute inter-object distances.")
                else:
                    st.info("Distance measurement is currently toggled off in the sidebar.")

    # TAB 2: Video & Stream Processing
    with tab2:
        st.subheader("Video File & Live Stream Object Detection with Color & Distance Tracking")
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
                            detect_color=enable_color,
                            measure_distance=enable_distance,
                            distance_threshold=float(dist_threshold),
                            distance_connect_mode=dist_mode,
                            annotate=True
                        )

                        all_video_detections.extend(res.detections)
                        counts_over_time.append({
                            "frame": frame_idx,
                            "count": res.total_count,
                            "fps": res.fps,
                            "closest_distance": res.closest_distance or 0.0
                        })

                        if res.annotated_image is not None:
                            rgb_frame = cv2.cvtColor(res.annotated_image, cv2.COLOR_BGR2RGB)
                            st_video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                        closest_info = f" | Closest Dist: {res.closest_distance}px" if res.closest_distance else ""
                        st_info_placeholder.text(f"Processing Frame {frame_idx}/{frame_count} | Objects: {res.total_count} | FPS: {res.fps}{closest_info}")
                        
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
                        detect_color=enable_color,
                        measure_distance=enable_distance,
                        distance_threshold=float(dist_threshold),
                        distance_connect_mode=dist_mode,
                        annotate=True
                    )

                    if res.annotated_image is not None:
                        rgb_frame = cv2.cvtColor(res.annotated_image, cv2.COLOR_BGR2RGB)
                        cam_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                    closest_info = f" | Closest Dist: {res.closest_distance}px" if res.closest_distance else ""
                    cam_info.markdown(f"**Live Feed FPS:** {res.fps} | **Current Object Count:** {res.total_count}{closest_info}")
                    frame_idx += 1
                cap.release()

    # TAB 3: Model Accuracy Benchmark
    with tab3:
        st.subheader("🎯 Model Accuracy & Performance Benchmarking")
        st.markdown("Evaluate standard object detection accuracy metrics (**mAP50**, **mAP50-95**, **Precision**, **Recall**, **F1-Score**) and inference latency across validation datasets.")

        bcol1, bcol2, bcol3 = st.columns([2, 2, 1])
        with bcol1:
            eval_model = st.selectbox("Model to Evaluate", available_models, index=available_models.index(selected_model) if selected_model in available_models else 0, key="eval_m")
        with bcol2:
            eval_dataset = st.text_input("Validation Dataset YAML", value="coco8.yaml", help="Path or name of dataset yaml (e.g. coco8.yaml, coco128.yaml, sample_data/data.yaml)")
        with bcol3:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            run_eval_btn = st.button("⚡ Run Accuracy Test", use_container_width=True)

        # Load existing report or run new evaluation
        report_file = os.path.join("outputs", "model_accuracy_report.json")
        eval_data = None

        if run_eval_btn:
            with st.spinner(f"Evaluating {eval_model} on {eval_dataset}..."):
                try:
                    eval_data = evaluate_model(model_path=eval_model, data_yaml=eval_dataset)
                    st.success("Accuracy evaluation completed successfully!")
                except Exception as ex:
                    st.error(f"Evaluation failed: {ex}")
        elif os.path.exists(report_file):
            try:
                import json
                with open(report_file, "r") as f:
                    eval_data = json.load(f)
            except Exception:
                pass

        if eval_data:
            ov = eval_data.get("overall_metrics", {})
            spd = eval_data.get("speed_benchmark_ms", {})

            # Metric Cards
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("mAP@50", f"{ov.get('mAP50', 0) * 100:.1f}%")
            mc2.metric("mAP@50-95", f"{ov.get('mAP50_95', 0) * 100:.1f}%")
            mc3.metric("Precision (P)", f"{ov.get('precision', 0) * 100:.1f}%")
            mc4.metric("Recall (R)", f"{ov.get('recall', 0) * 100:.1f}%")
            mc5.metric("F1-Score", f"{ov.get('f1_score', 0) * 100:.1f}%")

            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Inference Latency", f"{spd.get('inference', 0):.1f} ms")
            sc2.metric("Total Latency per Image", f"{spd.get('total_latency_per_image', 0):.1f} ms")
            sc3.metric("Effective Throughput", f"{spd.get('estimated_fps', 0)} FPS")

            # Per-class metrics table
            st.markdown("##### 📋 Per-Class Accuracy Breakdown")
            classes_breakdown = eval_data.get("per_class_breakdown", [])
            if classes_breakdown:
                df_cls = pd.DataFrame(classes_breakdown)
                # Format percentage columns
                for col in ["Precision", "Recall", "F1-Score", "mAP@50", "mAP@50-95"]:
                    if col in df_cls.columns:
                        df_cls[col] = df_cls[col].apply(lambda x: f"{x * 100:.1f}%")
                st.dataframe(df_cls, use_container_width=True)

            # Evaluation Curves & Plots
            st.markdown("##### 📈 Evaluation Curves & Confusion Matrix")
            curve_cols = st.columns(3)
            art_dir = eval_data.get("artifact_dir", "runs/evaluation")
            
            pr_path = os.path.join(art_dir, "BoxPR_curve.png")
            f1_path = os.path.join(art_dir, "BoxF1_curve.png")
            cm_path = os.path.join(art_dir, "confusion_matrix.png")

            if os.path.exists(pr_path):
                with curve_cols[0]:
                    st.caption("Precision-Recall Curve")
                    st.image(pr_path, use_container_width=True)
            if os.path.exists(f1_path):
                with curve_cols[1]:
                    st.caption("F1-Confidence Curve")
                    st.image(f1_path, use_container_width=True)
            if os.path.exists(cm_path):
                with curve_cols[2]:
                    st.caption("Confusion Matrix")
                    st.image(cm_path, use_container_width=True)

            # Download Report
            with open(report_file, "rb") as rf:
                st.download_button("📥 Download Full Accuracy JSON Report", rf, file_name="model_accuracy_report.json", mime="application/json")

    # TAB 4: Custom Model Training
    with tab4:
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

    # TAB 5: Analytics & Export
    with tab5:
        st.subheader("System Analytics & Configuration")
        st.json(config)

if __name__ == "__main__":
    main()
