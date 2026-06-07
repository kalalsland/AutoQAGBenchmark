# Copyright (c) Alibaba, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# --------------------------------------------------------
# 直接复用自 data-juicer (data_juicer/utils/registry.py)，
# 其本身改编自 modelscope。用作 AutoQAG 各 stage 的注册中心。
# --------------------------------------------------------


class Registry(object):
    """按仓库名注册模块的简单注册表。"""

    def __init__(self, name: str):
        self._name = name
        self._modules = {}

    @property
    def name(self):
        return self._name

    @property
    def modules(self):
        return self._modules

    def list(self):
        """列出当前注册表中的所有模块名。"""
        return list(self._modules.keys())

    def get(self, module_key):
        """按名取模块，找不到返回 None。"""
        return self._modules.get(module_key, None)

    def _register_module(self, module_name=None, module_cls=None, force=False):
        if module_name is None:
            module_name = module_cls.__name__

        if module_name in self._modules and not force:
            raise KeyError(f"{module_name} is already registered in {self._name}")

        self._modules[module_name] = module_cls
        module_cls._name = module_name

    def register_module(
        self, module_name: str = None, module_cls: type = None, force=False
    ):
        """把模块类注册到注册表，可作装饰器使用。

        Example:
            >>> STAGES = Registry("stages")
            >>> @STAGES.register_module("parse")
            >>> class ParseStage:
            >>>     pass
        """
        if not (module_name is None or isinstance(module_name, str)):
            raise TypeError(
                f"module_name must be either of None, str, got {type(module_name)}"
            )
        if module_cls is not None:
            self._register_module(
                module_name=module_name, module_cls=module_cls, force=force
            )
            return module_cls

        def _register(module_cls):
            self._register_module(
                module_name=module_name, module_cls=module_cls, force=force
            )
            return module_cls

        return _register


# 全局 stage 注册中心：每个流水线模块通过 @STAGES.register_module("xxx") 注册。
STAGES = Registry("stages")
