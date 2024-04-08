"""
General Video-to-Text Schema, including:
- Video captioning
- Video to text retrieval

Video datasets can be very large. For datasets with remote videos
('video_path' = video URL), ensure that the URL is publicly accessible
and the video is downloadable. Extra caution is needed, as the URL
might contain harmful and/or malicious files.
"""
import datasets

features = datasets.Features(
    {
        "id": datasets.Value("string"),
        "video_path": datasets.Value("string"),
        "text": datasets.Value("string"),
        "metadata": {
            "resolution": {
                "width": datasets.Value("int64"),
                "height": datasets.Value("int64"),
            },
            "duration": datasets.Value("float32"),
            "fps": datasets.Value("float32"),
        },
    }
)
