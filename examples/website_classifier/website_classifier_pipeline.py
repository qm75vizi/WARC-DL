import collections
import re
import resiliparse.parse.lang
import tensorflow as tf
from resiliparse.parse.encoding import detect_encoding, bytes_to_str
from urllib.parse import urlparse

from pipelines.html_pipeline import HTMLPipeline


def _strip_schema_org_annotation(html):
    return re.sub(r' itemscope itemtype="http:\/\/schema\.org+[^ \r\n>]*', '', html)


class WebsiteClassifierPipeline(HTMLPipeline):
    """
    This is an example html classification pipeline based on
    Here, we use this model to extract websites.
    """

    def __init__(self):
        self.domain_counter = collections.Counter()
        out_dir = "data/website_classifier/out/"
        max_content_length = 4000000
        super().__init__(out_dir=out_dir, max_content_length=max_content_length)

    def get_model(self):
        model = ""
        return model

    def predict(self, model_input, *args):
        prediction, *_ = super().predict(model_input)
        return ""

    def get_tokens_spec(self):
        return {'input_ids': tf.TensorSpec(shape=(None,), dtype=tf.int32),
                'attention_mask': tf.TensorSpec(shape=(None,), dtype=tf.int32)}

    def batch(self, dataset, batchsize):
        return dataset.padded_batch(batchsize, drop_remainder=True)

    def get_tokenizer(self):
        schema_newsarticle_annotation = "http://schema.org/NewsArticle"
        schema_blog_annotation = "http://schema.org/Blog"
        schema_event_annotation = "http://schema.org/Event"
        schema_product_annotation = "http://schema.org/Product"

        html = bytes_to_str(self, detect_encoding(self))

        if schema_product_annotation in self:
            tokenizer = {"html": _strip_schema_org_annotation(html), "label": "product"}
        if schema_newsarticle_annotation in self:
            tokenizer = {"html": _strip_schema_org_annotation(html), "label": "news"}
        if schema_blog_annotation in self:
            tokenizer = {"html": _strip_schema_org_annotation(html), "label": "blog"}
        if schema_event_annotation in self:
            tokenizer = {"html": _strip_schema_org_annotation(html), "label": "event"}

        def tokenizer_func(inp):
            return tokenizer(inp).data

        return tokenizer_func

    def get_distributed_filter(self):
        def distributed_filter(record):
            url = str(record.headers['WARC-Target-URI'])
            html_bytes = record.reader.read()
            html = bytes_to_str(html_bytes, detect_encoding(html_bytes))
            domain = urlparse(url).netloc

            if resiliparse.parse.lang.detect_fast(html)[0] != "en":
                return False
            if "lang=\"en\"" not in html:
                return False
            if "http://schema.org" not in html:
                return False
            if self.domain_counter.get(domain) >= 2000:
                return False
            return True

        return distributed_filter

    def filter(self, prediction, *args):
        return tf.reshape(prediction > .9, ())


if __name__ == "__main__":
    p = WebsiteClassifierPipeline()
    p.run()
