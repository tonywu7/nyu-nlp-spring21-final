# Copyright 2021 Tony Wu +https://github.com/tonywu7/
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

from __future__ import annotations

from collections.abc import MutableMapping, MutableSequence, MutableSet
from importlib.util import module_from_spec, spec_from_file_location
from typing import Any, overload

import simplejson as json


def _compose_mappings(*mappings):
    base = {}
    base.update(mappings[0])
    for m in mappings[1:]:
        for k, v in m.items():
            if k in base and type(base[k]) is type(v):
                if isinstance(v, MutableMapping):
                    base[k] = _compose_mappings(base[k], v)
                elif isinstance(v, MutableSet):
                    base[k] |= v
                elif isinstance(v, MutableSequence):
                    base[k].extend(v)
                else:
                    base[k] = v
            else:
                base[k] = v
    return base


class Settings(dict):
    @overload
    def __getitem__(self, k: str) -> Any:
        ...

    @overload
    def __getitem__(self, k: slice) -> Settings:
        ...

    def __getitem__(self, k):
        if isinstance(k, slice):
            return self.ns(k.start)
        return super().__getitem__(str(k).lower())

    def __setitem__(self, k, v) -> None:
        return super().__setitem__(str(k).lower(), v)

    def get(self, k, d=None):
        return super().get(str(k).lower(), d)

    def setdefault(self, k, d):
        return super().setdefault(str(k).lower(), d)

    def update(self, other):
        return super().update({str(k).lower(): v for k, v in other.items()})

    def normalize(self, check=str.islower, func=str.lower):
        convert = {}
        for k, v in self.items():
            if not check(k):
                convert[k] = v
        for k, v in convert.items():
            self[func(k)] = v
            del self[k]

    def from_pyfile(self, path):
        if not path.exists():
            return
        spec = spec_from_file_location('soundfinder.instance', path)
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.from_object(mod)

    def from_object(self, obj):
        keys = dir(obj)
        self.merge({k: getattr(obj, k) for k in keys if k.isupper()})

    def merge(self, other):
        d = _compose_mappings(self, other)
        self.clear()
        self.update(d)
        self.normalize()

    def ns(self, prefix: str) -> Settings:
        prefix = prefix.lower()
        return Settings({k.replace(prefix, '').lstrip('_'): v
                         for k, v in self.items()
                         if k.startswith(prefix)})

    # Following methods are from Scrapy
    def getbool(self, name, default=False):
        """
        Get a setting value as a boolean.

        ``1``, ``'1'``, `True`` and ``'True'`` return ``True``,
        while ``0``, ``'0'``, ``False``, ``'False'`` and ``None`` return ``False``.

        For example, settings populated through environment variables set to
        ``'0'`` will return ``False`` when using this method.

        :param name: the setting name
        :type name: string

        :param default: the value to return if no setting is found
        :type default: any
        """
        got = self.get(name, default)
        try:
            return bool(int(got))
        except ValueError:
            if got in ('True', 'true'):
                return True
            if got in ('False', 'false'):
                return False
            raise ValueError('Supported values for boolean settings '
                             "are 0/1, True/False, '0'/'1', "
                             "'True'/'False' and 'true'/'false'")

    def getint(self, name, default=0):
        """
        Get a setting value as an int.

        :param name: the setting name
        :type name: string

        :param default: the value to return if no setting is found
        :type default: any
        """
        return int(self.get(name, default))

    def getfloat(self, name, default=0.0):
        """
        Get a setting value as a float.

        :param name: the setting name
        :type name: string

        :param default: the value to return if no setting is found
        :type default: any
        """
        return float(self.get(name, default))

    def getlist(self, name, default=None):
        """
        Get a setting value as a list. If the setting original type is a list, a
        copy of it will be returned. If it's a string it will be split by ",".

        For example, settings populated through environment variables set to
        ``'one,two'`` will return a list ['one', 'two'] when using this method.

        :param name: the setting name
        :type name: string

        :param default: the value to return if no setting is found
        :type default: any
        """
        value = self.get(name, default or [])
        if isinstance(value, str):
            value = value.split(',')
        return list(value)

    def getdict(self, name, default=None):
        """
        Get a setting value as a dictionary. If the setting original type is a
        dictionary, a copy of it will be returned. If it is a string it will be
        evaluated as a JSON dictionary. In the case that it is a
        :class:`~scrapy.settings.BaseSettings` instance itself, it will be
        converted to a dictionary, containing all its current settings values
        as they would be returned by :meth:`~scrapy.settings.BaseSettings.get`,
        and losing all information about priority and mutability.

        :param name: the setting name
        :type name: string

        :param default: the value to return if no setting is found
        :type default: any
        """
        value = self.get(name, default or {})
        if isinstance(value, str):
            value = json.loads(value)
        return dict(value)
