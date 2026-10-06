# HNX-INT-049
# VisionGuard - Video Understanding and Temporal Reasoning

VisionGuard is a simple AI-based video analysis system developed for
HNX26PSI02: Video Understanding & Temporal Reasoning.

The system watches a video, detects objects and events, tracks them over
time, and answers questions about what happened and when it happened.

## What does this project do?

The main purpose of this project is to understand events in a video.

For example, we can ask:

- What happened after the truck arrived?
- When did the person enter the restricted area?
- How many times did the machine stop?
- What happened before the alarm?
- How long did a person stay in a particular area?

The system gives the answer along with the timestamp.

## Technologies Used

- Python
- OpenCV
- YOLO
- ByteTrack
- Streamlit
- NumPy
- Pandas
- PyTorch

## How it works

The basic working of our system is:

1. Upload a video.
2. Extract frames from the video.
3. Detect objects using YOLO.
4. Track the detected objects using ByteTrack.
5. Identify important events.
6. Store the events with their timestamps.
7. Process the user's question.
8. Compare the events based on time.
9. Display the answer with the timestamp.

### Example

Suppose the video contains these events:

```text
00:10 - Student 1 enters the classroom
00:25 - Student 2 enters the classroom
01:05 - Teacher enters the classroom
01:30 - Student 1 sits down
02:10 - Student 2 leaves the classroom
03:00 - Student 2 returns to the classroom
