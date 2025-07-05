# -*- coding: utf-8 -*-
# This file is auto-generated, don't edit it. Thanks.
import os
import sys

from typing import List

from alibabacloud_imageseg20191230.client import Client as imageseg20191230Client
from alibabacloud_credentials.client import Client as CredentialClient
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_imageseg20191230 import models as imageseg_20191230_models
from alibabacloud_tea_util import models as util_models
from alibabacloud_tea_util.client import Client as UtilClient


class Sample:
    def __init__(self):
        pass

    @staticmethod
    def create_client() -> imageseg20191230Client:
        """
        使用凭据初始化账号Client
        @return: Client
        @throws Exception
        """
        # 工程代码建议使用更安全的无AK方式，凭据配置方式请参见：https://help.aliyun.com/document_detail/378659.html。
        credential = CredentialClient()
        config = open_api_models.Config(
            credential=credential
        )
        # Endpoint 请参考 https://api.aliyun.com/product/imageseg
        config.endpoint = f'imageseg.cn-shanghai.aliyuncs.com'
        return imageseg20191230Client(config)

    @staticmethod
    def main(
        args: List[str],
    ) -> None:
        client = Sample.create_client()
        segment_hdbody_request = imageseg_20191230_models.SegmentHDBodyRequest(
            image_url='http://viapi-test.oss-cn-shanghai.aliyuncs.com/viapi-3.0domepic/imageseg/SegmentHDBody/SegmentHDBody1.jpg'
        )
        runtime = util_models.RuntimeOptions()
        try:
            # 复制代码运行请自行打印 API 的返回值
            client.segment_hdbody_with_options(segment_hdbody_request, runtime)
        except Exception as error:
            # 此处仅做打印展示，请谨慎对待异常处理，在工程项目中切勿直接忽略异常。
            # 错误 message
            print(error.message)
            # 诊断地址
            print(error.data.get("Recommend"))
            UtilClient.assert_as_string(error.message)

    @staticmethod
    async def main_async(
        args: List[str],
    ) -> None:
        client = Sample.create_client()
        segment_hdbody_request = imageseg_20191230_models.SegmentHDBodyRequest(
            image_url='http://viapi-test.oss-cn-shanghai.aliyuncs.com/viapi-3.0domepic/imageseg/SegmentHDBody/SegmentHDBody1.jpg'
        )
        runtime = util_models.RuntimeOptions()
        try:
            # 复制代码运行请自行打印 API 的返回值
            await client.segment_hdbody_with_options_async(segment_hdbody_request, runtime)
        except Exception as error:
            # 此处仅做打印展示，请谨慎对待异常处理，在工程项目中切勿直接忽略异常。
            # 错误 message
            print(error.message)
            # 诊断地址
            print(error.data.get("Recommend"))
            UtilClient.assert_as_string(error.message)


if __name__ == '__main__':
    Sample.main(sys.argv[1:])