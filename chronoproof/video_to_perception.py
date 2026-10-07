print("\n--- SCRIPT IS STARTING ---")
print("Loading libraries (this might take a few seconds)...")

import cv2
import csv
import math
import os
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO

print("Libraries loaded successfully!")

def get_seat_zone(x, y, width, height):
    row = "R1" if y < height / 2 else "R2"
    col = "C1" if x < width / 2 else "C2"
    return f"{row}{col}"

def generate_dynamic_action(kpts, objects_near, motion_level):
    """Procedurally generates a descriptive action string based on skeletal math and objects."""
    state_fragments = []
    
    # 1. Base Posture / Motion
    if motion_level > 0.25:
        state_fragments.append("moving actively")
    else:
        head_down = False
        if len(kpts) > 6 and kpts[0][1] > 0 and kpts[5][1] > 0 and kpts[6][1] > 0:
            nose_y = kpts[0][1]
            avg_shoulder_y = (kpts[5][1] + kpts[6][1]) / 2
            # If the nose drops down near or below the shoulders
            if nose_y > avg_shoulder_y - 15: 
                head_down = True
        
        if head_down:
            state_fragments.append("leaning down")
        else:
            state_fragments.append("sitting upright")

    # 2. Hand Position Analysis
    hands_raised = False
    hands_near_face = False
    
    if len(kpts) > 10:
        nose = kpts[0]
        l_sh, r_sh = kpts[5], kpts[6]
        l_wr, r_wr = kpts[9], kpts[10]
        
        avg_sh_y = (l_sh[1] + r_sh[1]) / 2 if (l_sh[1] > 0 and r_sh[1] > 0) else 0
        
        # Check if wrists are raised above shoulders
        if avg_sh_y > 0 and ((0 < l_wr[1] < avg_sh_y) or (0 < r_wr[1] < avg_sh_y)):
            hands_raised = True
            
        # Check if wrists are close to the face (nose)
        if nose[0] > 0:
            for wr in [l_wr, r_wr]:
                if wr[0] > 0:
                    dist = math.sqrt((wr[0] - nose[0])**2 + (wr[1] - nose[1])**2)
                    if dist < 45: # Pixel threshold for proximity
                        hands_near_face = True
                        break
                        
    if hands_near_face:
        state_fragments.append("hands near face")
    elif hands_raised:
        state_fragments.append("arms raised")

    # 3. Object Interaction
    if objects_near:
        # Remove duplicates if multiple of the same object are detected
        unique_objs = list(set(objects_near))
        obj_str = " & ".join(unique_objs)
        state_fragments.append(f"interacting with {obj_str}")

    return ", ".join(state_fragments)

def process_classroom_video(video_path: str, output_csv: str):
    print(f"\n[+] Opening video: {os.path.basename(video_path)}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[!] ERROR: Could not read '{video_path}'.")
        return

    print("[+] Loading YOLOv8m models into RTX 4050 VRAM...")
    pose_model = YOLO("yolov8m-pose.pt")
    obj_model = YOLO("yolov8m.pt")
    
    # Extract the dictionary of all 80 objects YOLO knows (e.g., bottle, tie, laptop, etc.)
    class_names = obj_model.names

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_step = max(1, int(fps * 0.5))
    
    frame_idx = 0
    previous_positions = {}
    
    active_events = {} # Track the unified action state for each person
    completed_events = []
    event_id_counter = 1

    print(f"[+] Processing video ({total_frames} frames)...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_step == 0:
            current_sec = round(frame_idx / fps, 1)

            # 1. Detect ALL objects in the frame
            obj_results = obj_model(frame, verbose=False, conf=0.45)[0]
            detected_objects = []
            if obj_results.boxes is not None:
                for box in obj_results.boxes:
                    cls_id = int(box.cls)
                    if cls_id != 0: # 0 is person, we only want inanimate objects
                        detected_objects.append({
                            "name": class_names[cls_id],
                            "coords": box.xyxy[0].cpu().numpy(),
                        })

            # 2. Track People and their Poses
            pose_results = pose_model.track(frame, persist=True, tracker="botsort.yaml", verbose=False, conf=0.35)[0]
            current_frame_states = {}

            if pose_results.boxes is not None and pose_results.boxes.id is not None:
                boxes = pose_results.boxes.xyxy.cpu().numpy()
                track_ids = pose_results.boxes.id.cpu().numpy()
                confs = pose_results.boxes.conf.cpu().numpy()
                kpts_all = pose_results.keypoints.xy.cpu().numpy() if pose_results.keypoints is not None else []

                for idx, (xyxy, t_id, person_conf) in enumerate(zip(boxes, track_ids, confs)):
                    p_id = f"S{int(t_id)}"
                    x_center = (xyxy[0] + xyxy[2]) / 2
                    y_center = (xyxy[1] + xyxy[3]) / 2
                    seat_zone = get_seat_zone(x_center, y_center, frame_width, frame_height)

                    # Calculate Motion
                    motion_level = 0.0
                    if p_id in previous_positions:
                        prev_x, prev_y = previous_positions[p_id]
                        dist = math.sqrt((x_center - prev_x)**2 + (y_center - prev_y)**2)
                        motion_level = min(dist / 50.0, 1.0)
                    previous_positions[p_id] = (x_center, y_center)

                    # Find all objects intersecting with this person
                    interacting_objs = []
                    for obj in detected_objects:
                        coords = obj["coords"]
                        if not (coords[2] < xyxy[0] or coords[0] > xyxy[2] or coords[3] < xyxy[1] or coords[1] > xyxy[3]):
                            interacting_objs.append(obj["name"])

                    # Procedurally generate the action string
                    kpts = kpts_all[idx] if idx < len(kpts_all) else []
                    action_string = generate_dynamic_action(kpts, interacting_objs, motion_level)
                    
                    current_frame_states[p_id] = {
                        "action": action_string,
                        "zone": seat_zone,
                        "conf": float(person_conf)
                    }

            # 3. State Management (Open/Close Events)
            for p_id in list(active_events.keys()):
                current_action = active_events[p_id]["action"]
                
                # If the person left the frame, OR their dynamically generated action changed
                if p_id not in current_frame_states or current_frame_states[p_id]["action"] != current_action:
                    event_data = active_events.pop(p_id)
                    end_sec = current_sec
                    duration = round(end_sec - event_data['start'], 1)
                    
                    # Log events lasting 1 second or more
                    if duration >= 1.0:
                        completed_events.append([
                            event_id_counter, p_id, event_data['action'], event_data['start'], 
                            end_sec, duration, event_data['zone'], round(event_data['conf'], 2)
                        ])
                        event_id_counter += 1

            # Start tracking new actions
            for p_id, data in current_frame_states.items():
                if p_id not in active_events:
                    active_events[p_id] = {
                        "action": data["action"],
                        "start": current_sec,
                        "zone": data["zone"],
                        "conf": data["conf"]
                    }

        frame_idx += 1

    # 4. Wrap up lingering events at the end of the video
    final_sec = round(total_frames / fps, 1)
    for p_id, event_data in active_events.items():
        duration = round(final_sec - event_data['start'], 1)
        if duration >= 1.0:
            completed_events.append([
                event_id_counter, p_id, event_data['action'], event_data['start'], 
                final_sec, duration, event_data['zone'], round(event_data['conf'], 2)
            ])
            event_id_counter += 1

    cap.release()

    print(f"\n[+] Writing {len(completed_events)} dynamically generated events to CSV...")
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Event_ID", "Person_ID", "Action", "Start_Sec", "End_Sec", "Duration_Sec", "Seat_Zone", "Confidence"])
        for event in completed_events:
            writer.writerow(event)

    print(f"[✓] Done! Output saved to: {os.path.abspath(output_csv)}")

if __name__ == "__main__":
    print("[*] Launching file picker dialog...")
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    video_file = filedialog.askopenfilename(
        title="Select Classroom Video",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
    )

    if not video_file:
        print("[!] No video selected. Exiting.")
    else:
        process_classroom_video(video_file, "perception.csv")