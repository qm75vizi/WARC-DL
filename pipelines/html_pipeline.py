import abc
import json
import os
import re
import tensorflow as tf
from collections import Counter
from fastwarc.warc import ArchiveIterator
from resiliparse.parse.encoding import detect_encoding, bytes_to_str
from urllib.parse import urlparse

from helpers import create_s3_client, get_file_stream
from pipelines.tools.passthrough_model import PassthroughModelPipeline


class HTMLPipeline(PassthroughModelPipeline):

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
            tf.TensorSpec(shape=(), dtype=tf.string))  # url

    def get_distributed_filter(self):
        acc_counter = self.acc_counter

        def distributed_filter(html, domain):
            if resiliparse.parse.lang.detect_fast(html)[0] != "en":
                return False
            if "lang=\"en\"" not in html:
                return False
            if "http://schema.org" not in html:
                return False
            if acc_counter.get(f"n_domain_{domain}") >= 2000:
                return False
            return True

        return distributed_filter

    def strip_schema_org_annotation(self, html):
        return re.sub(r' itemscope itemtype="http:\/\/schema\.org+[^ \r\n>]*', '', html)

    def get_annotation(self):

        def annotation(html):
            schema_newsarticle_annotation = "http://schema.org/NewsArticle"
            schema_blog_annotation = "http://schema.org/Blog"
            schema_event_annotation = "http://schema.org/Event"
            schema_product_annotation = "http://schema.org/Product"

            if schema_product_annotation in html:
                annotation = "product"
            if schema_newsarticle_annotation in html:
                annotation = "news"
            if schema_blog_annotation in html:
                annotation = "blog"
            if schema_event_annotation in html:
                annotation = "event"
            return annotation

        return annotation

    def get_generator_factory(self):
        acc_counter = self.acc_counter
        max_content_length = self.max_content_length
        distributed_filter = self.get_distributed_filter()
        annotator = self.get_annotation()
        stripped_html = self.strip_schema_org_annotation()
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
                    if record.headers['WARC-Type'] == 'response' and record.content_length >= 128:
                        content_type = str(record.http_content_type).lower()
                        if content_type.startswith("text/html"):
                            url = str(record.headers['WARC-Target-URI'])
                            domain = urlparse(url).netloc
                            acc_counter.add(Counter({f"n_domain_{domain}": 1}))
                            html_bytes = record.reader.read()
                            html = bytes_to_str(html_bytes, detect_encoding(html_bytes))

                            if not distributed_filter(html, domain):
                                acc_counter.add(Counter({"n_distributed_filter_not_passed": 1}))
                                continue

                            yield stripped_html(html), annotator(html), url
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
        print(url, html, annotation)
        with open(self.out_dir, "a") as f:
            f.write(
                json.dumps(
                    {
                        'url': url,
                        'annotation': annotation,
                        'html': html,
                    },
                    ensure_ascii=False) + '\n'
            )


if __name__ == "__main__":
    p = HTMLPipeline(out_dir="data/website_classifier/out/", max_content_length=4000000)
    p.run()
