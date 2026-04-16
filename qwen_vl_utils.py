from __future__ import annotations

from typing import Any


def process_vision_info(messages: list[dict[str, Any]]):
    images: list[Any] = []
    videos: list[Any] = []
    for message in messages:
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            item_type = item.get("type")
            if item_type == "image":
                image = item.get("image")
                if image is not None:
                    images.append(image)
            elif item_type == "video":
                video = item.get("video")
                if video is not None:
                    videos.append(video)
    return images, videos
