import abc
import json
import os
import re
from collections import Counter
from urllib.parse import urlparse

import pandas as pd
import tensorflow as tf
from fastwarc.warc import ArchiveIterator
from helpers import create_s3_client, get_file_stream
from resiliparse.parse.encoding import bytes_to_str, detect_encoding
from resiliparse.parse.lang import detect_fast

from pipelines.tools.passthrough_model import ExportDatasetPipeline


class HTMLPipeline(Pipeline, abc.ABC):
    def __init__(self, out_dir, max_content_length):
        self.out_dir = out_dir
        if self.out_dir is not None:
            os.makedirs(self.out_dir, exist_ok=True)
        self.max_content_length = max_content_length

        super().__init__()

    def get_signature(self):
        return (
            tf.TensorSpec(shape=(), dtype=tf.string),  # plain html text
            tf.TensorSpec(shape=(), dtype=tf.string),  # annotation
            tf.TensorSpec(shape=(), dtype=tf.string),  # domain
        )  

    def get_distributed_filter(self):
        acc_counter = self.acc_counter

        def distributed_filter(html, domain):
            if detect_fast(html)[0] != "en":
                return False
            if 'lang="en"' not in html:
                return False
            if "http://schema.org" not in html:
                return False
            if acc_counter.get(f"n_domain_{domain}") >= 2000:
                return False
            return True

        return distributed_filter

    def get_strip_schema_org_annotation(self):
        def strip_schema_org_annotations(html):
            # remove itemscope and itemtype
            html = re.sub(r' itemscope itemtype="http:\/\/schema\.org+[^ \r\n>]*', "", html)
            # remove itemprop
            html = re.sub(r' itemprop=["\']\w*["\']', "", html)
            return html

        return strip_schema_org_annotations

    def get_annotation(self):
        def annotation(html):
            # schema_map to df
            schema_map_df = pd.read_csv("resources/schema_map.csv")
            # list of all existing schema_org annotations in html
            existing_schemas = re.findall(r'http:\/\/schema\.org+[^ \r\n>"\']*', html)
            # iterate through list of schemas to find label in data frame
            for schema in existing_schemas:
                annotation = schema_map_df[
                    schema_map_df["schema"] == re.sub(r"http:\/\/schema\.org\/", "", schema)
                ].label.values
                if annotation.size > 0:
                    return annotation[0]

        return annotation

    def get_generator_factory(self):
        acc_counter = self.acc_counter
        max_content_length = self.max_content_length
        distributed_filter = self.get_distributed_filter()
        annotator = self.get_annotation()
        stripped_html = self.get_strip_schema_org_annotation()
        AWS_ACCESS_KEY_ID = self.AWS_ACCESS_KEY_ID
        AWS_SECRET = self.AWS_SECRET
        ENDPOINT_URL = self.ENDPOINT_URL

        def generator_factory(file_identifier):
            s3_client = create_s3_client(AWS_ACCESS_KEY_ID, AWS_SECRET, ENDPOINT_URL)
            stream = get_file_stream(s3_client, file_identifier)
            for record in ArchiveIterator(stream, max_content_length=max_content_length):
                try:
                    if record.headers is None:
                        acc_counter.add(Counter({"n_record_headers_none": 1}))
                        continue
                    if record.http_headers is None:
                        acc_counter.add(Counter({"n_http_headers_none": 1}))
                        continue
                    if record.headers["WARC-Type"] == "response" and record.content_length >= 128:
                        content_type = str(record.http_content_type).lower()
                        if content_type.startswith("text/html"):
                            url = str(record.headers["WARC-Target-URI"])
                            domain = urlparse(url).netloc
                            acc_counter.add(Counter({f"n_domain_{domain}": 1}))
                            html_bytes = record.reader.read()
                            html = bytes_to_str(html_bytes, detect_encoding(html_bytes))

                            if not distributed_filter(html, domain):
                                acc_counter.add(Counter({"n_distributed_filter_not_passed": 1}))
                                continue

                            yield stripped_html(html), annotator(html), domain
                            acc_counter.add(Counter({"n_node_results": 1}))
                            print("print acc in generator_factory", acc_counter)

                        else:
                            acc_counter.add(Counter({"n_wrong_content_type": 1}))
                    else:
                        acc_counter.add(Counter({"n_wrong_warc_type": 1}))
                except:
                    acc_counter.add(Counter({"n_unhandled_record_exceptions": 1}))
                    continue
            acc_counter.add(Counter({"n_finished_warc_files": 1}))

        return generator_factory

    def export(self, html, annotation, url):
        with open(self.out_dir, "a") as f:
            f.write(
                json.dumps(
                    {
                        "domain": domain,
                        "annotation": annotation,
                        "html": html,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


if __name__ == "__main__":
    p = HTMLPipeline(out_dir="data/website_classifier/out/", max_content_length=4000000)
    p.run()
