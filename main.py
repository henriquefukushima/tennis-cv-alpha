from utils import (read_video,
                   save_video)
from trackers import (PlayerTracker,
                        BallTracker)
from courtline_detector import CourtLineDetector

def main():
    # Read Video
    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)

    # Track Players 
    player_tracker = PlayerTracker(model_path="models/yolov8x.pt")
    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="tracker_stubs/player_detections.pkl")
    # Track Ball
    ball_tracker = BallTracker(model_path="models/yolov5n_last.pt")
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                 read_from_stub=True,
                                                 stub_path="tracker_stubs/ball_detections.pkl")
    ball_detections = ball_tracker.interpolate_ball_detections(ball_detections)

    # Court Line Detector
    court_model_path = "models/court_kps_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # Choose and Filter Players
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    # Draw Bounding Boxes
    video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    video_frames = ball_tracker.draw_bboxes(video_frames, ball_detections)
    # Draw Court Keypoints
    video_frames = court_line_detector.draw_keypoints_on_video(video_frames, court_keypoints)

    # Save Output Video
    save_video(video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()
