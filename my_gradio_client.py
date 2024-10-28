import os.path
import sqlite3
import time
from functools import lru_cache
from typing import List

from gradio_client import Client

db_path = "ktem_app_data/user_data/sql.db"


@lru_cache(1)
def get_all_file_ids(db_path: str, indices: List[int] | None = None):
    assert os.path.isfile(db_path)

    file_ids: List[str] = []

    conn = sqlite3.connect(db_path)

    try:
        c = conn.cursor()
        if indices is None:
            # Get all fileIndex
            c.execute("SELECT * FROM ktem__index")
            indices = [each[0] for each in c.fetchall()]

        # For each index get all file_ids
        for i in indices:
            table_name = f"index__{i}__source"
            c.execute(
                f"SELECT * FROM {table_name}",
            )
            file_ids += [each[0] for each in c.fetchall()]
    finally:
        conn.close()

    return file_ids


file_ids = get_all_file_ids(db_path)

print(f"File_ids: {file_ids}")

client = Client("http://localhost:7860/")
# client = Client("https://kotaemon.sv.nequal.cn/")
# client = Client("https://26826adcd28b3e4414.gradio.live")

job = client.submit(
    [
        [
            # "2022年出生人口",
            # "2023年第三季度GDP增速是多少",
            "在关于中国消费者兴趣圈层的调研中，某美妆品牌高相关目标圈层覆盖人群数量排名第二、第三、第四的是哪些?",
            None,
        ]
    ],
    "select",
    file_ids,
    api_name="/chat_fn",
)

# while True:
    # if not job.done():
        # time.sleep(0.2)
    # else:
        # break


# print(job.outputs()[-1][0][0][1])
# print(job.result()[0][0][1])
print(job.result())
