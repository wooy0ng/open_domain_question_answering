import os
import json
import pandas as pd
import numpy as np
import pickle as pkl
import faiss
from tqdm import tqdm
from utils import Spinner
from typing import Optional, Union, Tuple, List
from transformers import PreTrainedTokenizer
from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer

spinner = Spinner()


class SparseRetrieval:
    def __init__(
        self,
        tokenize_fn: PreTrainedTokenizer,
        data_path: Optional[str],
        context_path: Optional[str],
    ):
        """
        Args:
            tokenize_fn:
                PreTrainedTokenizer의 tokenize 메소드
            data_path:
                데이터가 위치한 path
            context_path:
                Passage들이 묶여있는 파일 명

        description:
            Passage 파일을 불러오고 TfidfVectorizor를 선언하는 기능 수행
        """

        self.data_path = data_path
        self.context_path = context_path
        with open(self.context_path, "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # 문장 중복 제거
        self.ids = list(range(len(self.contexts)))

        # TFIDF 변환
        self.tfidf = TfidfVectorizer(
            tokenizer=tokenize_fn,
            ngram_range=(1, 2),
            max_features=50000,
        )

        self.passage_embedding = None  # get_sparse_embedding() 함수가 생성됨
        self.indexer = None  # build_faiss() 함수가 생성됨

    def get_sparse_embedding(self) -> None:
        """
        Description:
            Passage Embedding을 만들고 TFIDF와 Embedding을 pkl로 저장함.
            만약 미리 저장된 파일이 있는 경우 저장된 pkl 파일을 불러옴.
        """

        pkl_name = "sparse_embedding.bin"
        tfidf_name = "tfidf.bin"
        embed_path = os.path.join(self.data_path, pkl_name)
        tfidfv_path = os.path.join(self.data_path, tfidf_name)

        if os.path.isfile(embed_path) and os.path.isfile(tfidfv_path):
            # 존재할 경우
            with open(embed_path, "rb") as f:
                self.passage_embedding = pkl.load(f)
            with open(tfidfv_path, "rb") as f:
                self.tfidf = pkl.load(f)
        else:
            # 존재하지 않을 경우
            spinner.start(desc="tfidf")
            self.passage_embedding = self.tfidf.fit_transform(self.contexts)
            spinner.stop()

            with open(embed_path, "wb") as f:
                pkl.dump(self.passage_embedding, f)
            with open(tfidfv_path, "wb") as f:
                pkl.dump(self.tfidf, f)
        return

    
    def get_faiss_indexer(self, num_clusters: int = 64) -> None:
        """
        Args:
            num_clusters:
                build 시, 벡터의 클러스터 개수를 지정

        Description:
            미리 Passage Embedding된 값을
            FAISS indexer에 fitting시켜 놓는 작업을 수행.
        """
        indexer_name = f"faiss_clusters{num_clusters}.index"
        indexer_path = os.path.join(self.data_path, indexer_name)
        if os.path.isfile(indexer_path):
            # Load saved faiss indexer
            self.indexer = faiss.read_index(indexer_path)
        else:
            passage_embedding = self.passage_embedding.astype(np.float32).toarray()
            embed_dim = passage_embedding.shape[-1]

            spinner.start(desc="faiss indexing")
            quantizer = faiss.IndexFlatL2(embed_dim)
            self.indexer = faiss.IndexIVFScalarQuantizer(
                quantizer, quantizer.d, num_clusters, faiss.METRIC_L2
            )
            self.indexer.train(passage_embedding)
            self.indexer.add(passage_embedding)
            spinner.stop()

            faiss.write_index(self.indexer, indexer_path)
            print("save faiss indexer.")

        return

    def retrieve(
        self,
        query_or_dataset: Union[str, Dataset],
        top_k: Optional[int] = 1,
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """
        Args:
            query_or_dataset:
                string이나 Dataset으로 이루어진 Query를 받는다.
                string으로 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구한다.
                Dataset의 형태는 query를 포함한 Dataset을 받는다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구한다.
            top_k:
                상위 몇 개의 passage를 사용할 것인지 지정한다.

        Returns:
            1개의 query를 받는 경우 -> Tuple[List, List]
            여러개의 query를 받는 경우 -> pd.DataFrame

        """

        assert (
            self.passage_embedding is not None
        ), "get_sparse_embedding()을 먼저 수행해야 합니다."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=top_k)
            for idx in range(top_k):
                print(f"Top-{idx+1} passage with score {doc_scores[idx]:.4f}")
                print(self.contexts[doc_indices[idx]])

            return (
                doc_scores,
                [self.contexts[doc_indices[idx]] for idx in range(top_k)],
            )

        elif isinstance(query_or_dataset, Dataset):
            # relatieve한 Passage를 pd.DataFrame으로 변환
            total = []
            doc_scores, doc_indices = self.get_relevant_doc_bulk(
                query_or_dataset["question"], k=top_k
            )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="[+] Sparse retrieval")
            ):
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context": "".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground truth context와 answer도 반환
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)
            cqas = pd.DataFrame(total)
            return cqas
        return

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        """
        Args:
            query:
                하나의 Query를 받음
            k:
                상위 몇 개의 passage를 받을 지 결정

        Description:
            vocab에 없는 이상한 단어로 query하는 경우 (ex. 뙯뚪)
        """
        query_vector = self.tfidf.transform([query])
        assert np.sum(query_vector) != 0, "query에 vectorizer의 vocab에 없는 단어만 존재할 경우 발생"

        spinner.start(desc="inner product")
        result = query_vector * self.passage_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()
        spinner.stop()

        sorted_result = np.argsort(result.squeeze())[::-1]
        doc_scores = result.squeeze()[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]

        return doc_scores, doc_indices

    def get_relevant_doc_bulk(self, queries: List, k: Optional[int] = 1):
        """
        Args:
            queries:
                여러 개의 Query를 받음
            k:
                상위 몇 개의 Passage를 받을 지 결정

        Description:
            vocab에 없는 이상한 단어로 query하는 경우 (ex. 뙯뚪)

        Return:
            각 query와 Passage를 비교하여 가장 유사도가 높은 score와 그 score의 index를 return
        """
        query_vectors = self.tfidf.transform(queries)
        assert np.sum(query_vectors) != 0, "query에 vectorizer의 vocab에 없는 단어만 존재할 경우 발생"

        spinner.start(desc="inner product")
        result = query_vectors * self.passage_embedding.T  # inner product (similarity)
        if not isinstance(result, np.ndarray):
            result = result.toarray()
        spinner.stop()

        doc_scores, doc_indices = [], []
        for idx, _ in enumerate(
            tqdm(
                result,
                desc="[+] calculate doc_scores & doc_indices",
                total=result.shape[0],
            )
        ):
            sorted_result = np.argsort(result[idx, :])[::-1]
            doc_scores.append(
                result[idx, :][sorted_result].tolist()[:k]
            )  # 가장 높은 similarity 순으로 score 저장
            doc_indices.append(
                sorted_result.tolist()[:k]
            )  # 가장 높은 similarity 순으로 index 저장

        return doc_scores, doc_indices
