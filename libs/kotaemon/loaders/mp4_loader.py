import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from decouple import config
from llama_index.core.readers.base import BaseReader

from kotaemon.base import Document

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
# ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

DEFAULT_VLM_ENDPOINT = (
    "{0}/chat/completions?".format(
        config("AZURE_OPENAI_ENDPOINT", default=""),
    )
)


class Mp4Reader(BaseReader):
    """Read Mp4 .
    Be able to extract text, audio, and figure with high accuracy

    Example:
        ```python
        >> from kotaemon.loaders import AdobeReader
        >> reader = AdobeReader()
        >> documents = reader.load_data("path/to/pdf")
        ```
    Args:
        endpoint: URL to the Vision Language Model endpoint. If not provided,
        will use the default `kotaemon.loaders.adobe_loader.DEFAULT_VLM_ENDPOINT`

        max_figures_to_caption: an int decides how many figured will be captioned.
        The rest will be ignored (are indexed without captions).
    """

    def __init__(
        self,
        vlm_endpoint: Optional[str] = None,
        max_frames_to_caption: int = 300,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Init params"""
        super().__init__(*args)
        self.vlm_endpoint = vlm_endpoint or DEFAULT_VLM_ENDPOINT
        self.max_frames_to_caption = max_frames_to_caption

    def load_data(
        self, file: Path, extra_info: Optional[Dict] = None, **kwargs
    ) -> List[Document]:
        """Load data by calling to the Adobe's API

        Args:
            file (Path): Path to the PDF file

        Returns:
            List[Document]: list of documents extracted from the PDF file,
                includes 3 types: text, table, and image

        """
        from .utils.adobe import (
            generate_figure_captions,
            parse_figure_paths,
            parse_table_paths,
        )
        from .utils.video import (
            extract_video_keyframes,
            extract_video_audio,
            split_and_transcribe_audio,
            chinese_to_pinyin,
        )

        filename = chinese_to_pinyin(file.name.split('.')[0])
        filepath = str(Path(file).resolve())
        print("extracting audio...")
        results_path = extract_video_audio(filename, filepath)
        print("results_path: " + results_path)

        if not os.path.exists(results_path):
            logger.exception("Fail to extract the audio.")
            return []
        
        audio_text = split_and_transcribe_audio(filename, results_path)
        print(f"audio_text: {audio_text}")

        # Wrap elements with Document
        documents = []
        documents.append(
            Document(
                text=audio_text,
                metadata={
                    "file_name": filename,
                    "file_path": filepath,
                    "type": "video",
                    **(extra_info if extra_info is not None else {}),
                },
            )
        )

        print("extracting keyframes...")
        keyframe_paths = extract_video_keyframes(filename, filepath)

        figures = []
        for page_number, keyframe_path in enumerate(keyframe_paths):
            figure_content = parse_figure_paths([Path(keyframe_path)])
            if not figure_content:
                continue
            figures.append([page_number, figure_content])


        print(f"{len(figures)} 帧需要识别")
        # get figure caption using GPT-4V
        figure_captions = generate_figure_captions(
            self.vlm_endpoint,
            [item[1] for item in figures],
            self.max_frames_to_caption,
        )

        # figure elements
        for page_number, figure_caption in enumerate(figure_captions):
            print(f"frame_caption: {figure_caption}")
            documents.append(
                Document(
                    text=figure_caption,
                    metadata={
                        "type": "video",
                        "page_label": page_number,
                        "file_name": filename,
                        "file_path": filepath,
                        **(extra_info if extra_info is not None else {}),
                    },
                    metadata_template="",
                    metadata_seperator="",
                )
            )
        return documents
