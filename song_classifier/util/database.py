# Copyright 2021 Tony Wu
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

import logging
import time
import uuid
from abc import ABC, abstractmethod
from datetime import date, datetime, timezone
from functools import wraps
from sqlite3 import Connection as SQLite3Connection
from typing import (Any, Dict, List, Optional, Tuple, Type, TypeVar, Union,
                    overload)
from urllib.parse import urlencode

import simplejson as json
import udatetime
from sqlalchemy import MetaData, Table, create_engine, event, types
from sqlalchemy.engine import Connection, Engine, Result
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import (Mapper, Session, declarative_base, declared_attr,
                            relationship, scoped_session, sessionmaker)
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.schema import Column, ForeignKey, Index
from sqlalchemy.sql.expression import FunctionElement, column, select, table
from sqlalchemy.sql.functions import count
from sqlalchemy.types import CHAR, INTEGER, TypeDecorator

metadata = MetaData(
    naming_convention={
        'ix': 'ix_%(table_name)s_%(column_0_N_name)s',
        'uq': 'uq_%(table_name)s_%(column_0_N_name)s',
        'ck': 'ck_%(table_name)s_%(column_0_N_name)s',
        'fk': 'fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s',
        'pk': 'pk_%(table_name)s',
    },
)
RESTRICT = 'RESTRICT'
CASCADE = 'CASCADE'

R = TypeVar('R')
T = TypeVar('T', bound='Identity')
U = TypeVar('U', bound='Identity')

InferrableSelectable = Union[Type[T], InstrumentedAttribute]
MappedSelectable = Union[Type[T], Mapper]

SessionFactory = None

__version__ = Table('__version__', metadata, Column('version', types.String(), primary_key=True))


def exclusive_to(classname: str, default=()):
    def wrapper(f):
        @wraps(f)
        def wrapped(cls):
            if cls.__name__ == classname:
                return f(cls)
            return default
        return wrapped
    return wrapper


class UUIDType(TypeDecorator):
    impl = CHAR

    @property
    def python_type(self):
        return uuid.UUID

    def load_dialect_impl(self, dialect):
        return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value: Optional[uuid.UUID | str], dialect) -> Optional[str]:
        if value is None:
            return None
        return str(value)

    def process_result_value(self, value: Optional[str], dialect) -> Optional[uuid.UUID]:
        if value is None:
            return None
        if not isinstance(value, uuid.UUID):
            return uuid.UUID(value)
        return value


class unixepoch(FunctionElement):
    type = INTEGER


@compiles(unixepoch, 'sqlite')
def sqlite_utcnow(element, compiler, **kwargs):
    return "CAST((julianday('now') - 2440587.5) * 86400000 AS INTEGER)"


def ensure_datetime(o):
    if o is None:
        return None
    if isinstance(o, datetime):
        return o
    if isinstance(o, date):
        return datetime.combine(o, datetime.min.time(), tzinfo=timezone.utc)
    if isinstance(o, (int, float)):
        return udatetime.fromtimestamp(o, tz=udatetime.TZFixedOffset(0))
    if isinstance(o, date):
        o = datetime.combine(o, datetime.min.time())
    try:
        return udatetime.from_string(o)
    except Exception:
        return o


class TimestampType(TypeDecorator):
    impl = INTEGER

    @property
    def python_type(self):
        return datetime

    def process_bind_param(self, value: Optional[int | float | datetime], dialect) -> Optional[float]:
        value = ensure_datetime(value)
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return int(value * 1000)
        try:
            return int(value.astimezone(timezone.utc).timestamp() * 1000)
        except AttributeError:
            pass
        raise TypeError(f'expected datetime.datetime object, not {type(value)}')

    def process_result_value(self, value: Optional[int], dialect) -> Optional[datetime]:
        if value is None:
            return None
        return udatetime.utcfromtimestamp(value / 1000)


@event.listens_for(Engine, 'connect')
def sqlite_features(conn, conn_record):
    if isinstance(conn, SQLite3Connection):
        with conn:
            conn.execute('PRAGMA foreign_keys=ON;')
            conn.execute('PRAGMA journal_mode=WAL;')


Base = declarative_base(metadata=metadata)


class Identity(Base):
    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not getattr(self, 'uuid4', None):
            self.uuid4 = uuid.uuid4()

    @classmethod
    def _parent_mapper(cls):
        for c in cls.mro()[1:]:
            if issubclass(c, Identity):
                return c

    @classmethod
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        supercls = cls._parent_mapper()
        cls.id: int = Column(types.Integer(), ForeignKey(supercls.id, ondelete=CASCADE, onupdate=RESTRICT), primary_key=True)

    id: int = Column(types.Integer(), primary_key=True, autoincrement=True)
    uuid4: uuid.UUID = Column(UUIDType())
    model: str = Column(types.String())
    ctime: datetime = Column(TimestampType(), nullable=False, server_default=unixepoch())

    @declared_attr
    def __mapper_args__(cls):
        args = {
            'polymorphic_identity': cls.__name__,
        }
        if cls.__name__ == 'Identity':
            args['polymorphic_on'] = cls.model
        return args

    @declared_attr
    @exclusive_to('Identity')
    def __table_args__(cls):
        return (
            Index(None, 'id', 'uuid4', unique=True),
            Index(None, 'uuid4', unique=True),
        )

    @property
    def ident(self) -> Identity:
        pk = self.id
        if not pk and not self.uuid4:
            raise ValueError
        if not pk:
            return self.uuid4
        return pk

    def discriminator(self, sep='#') -> str:
        return f'{type(self).__name__}{sep}{self.ident}'

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__reflection__.columns}

    def __str__(self):
        return self.discriminator()

    def __repr__(self) -> str:
        return f'<{self.discriminator()} at {hex(id(self))}>'


class BundleABC(ABC):
    def __init__(self, path, *, echo=False, readonly=False, thread_safe=True):
        db_url = f'sqlite:///{path}'
        params = {}
        if readonly:
            params['mode'] = 'ro'
        if not thread_safe:
            params['check_same_thread'] = False
        if params:
            db_url = f'{db_url}?{urlencode(params)}'

        engine = create_engine(db_url, echo=echo, json_serializer=JSON_ENCODER.encode)

        self._metadata: MetaData = metadata
        self._engine: Engine = engine
        set_session(scoped_session(sessionmaker(bind=engine, autoflush=False, future=True)))

        self._init_logging()
        self._init_events()
        self._verify_version()

        metadata.create_all(engine)
        self._init_version()

    def _init_logging(self):
        self.log = logging.getLogger('rdb.bundle')
        self.log_timing = logging.getLogger('rdb.bundle.timing')
        logging.getLogger('sqlalchemy.engine.Engine').handlers.clear()

    @property
    @abstractmethod
    def version(self) -> str:
        raise NotImplementedError

    def _verify_version(self):
        stmt = __version__.select()
        try:
            row = self.execute(stmt).fetchone()
            ver = row and row[0]
        except OperationalError:
            ver = None
        if not ver:
            stmt = select([count(column('name'))]).select_from(table('sqlite_master')).where(column('type') == 'table')
            table_count = self.execute(stmt).fetchone()[0]
            if table_count:
                raise DatabaseNotEmptyError()
            return
        if ver != self.version:
            raise DatabaseVersionError(ver, self.version)

    def _init_version(self):
        self.execute('INSERT OR REPLACE INTO __version__ (version) VALUES (?);', (self.version,))
        self.commit()

    def _init_events(self):
        event.listen(Engine, 'before_cursor_execute', self._on_before_cursor_execute)
        event.listen(Engine, 'after_cursor_execute', self._on_after_cursor_execute)

    @overload
    def __getitem__(self, key: int | str | uuid.UUID) -> Optional[T]:
        ...

    @overload
    def __getitem__(self, key: Tuple[Type[U], int | str | uuid.UUID]) -> Optional[U]:
        ...

    def __getitem__(self, key):
        if isinstance(key, tuple):
            model, ident = key
        else:
            model = None
            ident = key
        if ident is None:
            return None
        if isinstance(ident, str):
            ident = uuid.UUID(ident)
        if model is None:
            model = Identity
        if isinstance(ident, uuid.UUID):
            return self.session.query(model).filter_by(uuid4=ident).one()
        return self.session.get(model, ident)

    def id(self, uid: uuid.UUID | str) -> Optional[int]:
        uid = str(uid)
        return self.query(Identity.id).filter(Identity.uuid4 == uid).scalar()

    def uid(self, id_: int) -> Optional[int]:
        return self.query(Identity.uuid4).filter(Identity.id == id_).scalar()

    def ids(self, uids: List[uuid.UUID]) -> Dict[uuid.UUID, int]:
        uids = [str(u) for u in uids]
        return dict(self.query(Identity.uuid4, Identity.id).filter(Identity.uuid4.in_(uids)).all())

    def uids(self, ids: List[int]) -> Dict[int, uuid.UUID]:
        return dict(self.query(Identity.id, Identity.uuid4).filter(Identity.id.in_(ids)).all())

    def _on_before_cursor_execute(self, conn, cursor, statement,
                                  parameters, context, executemany):
        conn.info.setdefault('query_start_time', []).append(time.time())

    def _on_after_cursor_execute(self, conn, cursor, statement,
                                 parameters, context, executemany):
        total = time.time() - conn.info['query_start_time'].pop(-1)
        self.log_timing.debug(f'Total Time: {total}')

    @property
    def session(self) -> Session:
        return get_session()

    @property
    def conn(self) -> Connection:
        return self.session.connection()

    def execute(self, stmt, *args, **kwargs) -> Result:
        return self.conn.execute(stmt, *args, **kwargs)

    @property
    def query(self):
        return self.session.query

    @property
    def flush(self):
        return self.session.flush

    @property
    def commit(self):
        return self.session.commit

    @property
    def rollback(self):
        return self.session.rollback

    @property
    def __contains__(self):
        return self.session.__contains__

    @property
    def __iter__(self):
        return self.session.__iter__


def get_session(**kwargs) -> Session:
    return SessionFactory(**kwargs)


def set_session(scs):
    global SessionFactory
    SessionFactory = scs


def del_session():
    SessionFactory.remove()


class DatabaseVersionError(ValueError):
    def __init__(self, identified, expected):
        super().__init__(f'Database version "{identified}" is different from supported version "{expected}".')


class DatabaseNotEmptyError(ValueError):
    def __init__(self):
        super().__init__('Database is not empty.')


def json_format(o):
    if isinstance(o, datetime):
        return o.isoformat()
    return str(o)


JSON_ENCODER = json.JSONEncoder(for_json=True, iterable_as_array=True, default=json_format)


class Relationship:
    cached = {}
    lookup: Dict[Tuple[Type[T], str], Table] = {}

    @classmethod
    def get_table_name(cls, src: str, dst: str) -> str:
        return f'_g_{src}_{dst}'.lower()

    @classmethod
    def register(cls, fwd: str, rev: str, join: Table | Type[T]):
        cls.cached[join.name] = join
        if not isinstance(join, Table):
            join = join.__table__
        cls.lookup[join, fwd] = 0
        cls.lookup[join, rev] = 1

    @classmethod
    def table(cls, rel: InferrableSelectable) -> Table:
        if isinstance(rel, InstrumentedAttribute):
            return rel.parent.get_property(rel.key).secondary

    @classmethod
    def direction(cls, rel: InferrableSelectable) -> int:
        if isinstance(rel, InstrumentedAttribute):
            return cls.lookup[cls.table(rel), rel.key]

    @classmethod
    def create_table(cls, name):
        return Table(
            name, metadata,
            Column('id', types.Integer(), primary_key=True, autoincrement=True),
            Column('src', types.Integer(), ForeignKey(Identity.id, ondelete=CASCADE, onupdate=RESTRICT), nullable=False),
            Column('dst', types.Integer(), ForeignKey(Identity.id, ondelete=CASCADE, onupdate=RESTRICT), nullable=False),
            Index(None, 'src', 'dst', unique=True),
        )

    @classmethod
    def two_way(cls, tables: Dict[str, str | Type[T]], **kwargs):
        (src_attr, src_model), (dst_attr, dst_model) = sorted(tables.items())
        table_name = cls.get_table_name(src_attr, dst_attr)
        join_table = cls.cached.get(table_name)
        if join_table is None:
            join_table = cls.create_table(table_name)
            cls.register(src_attr, dst_attr, join_table)
        return {
            dst_attr: relationship(
                dst_model, back_populates=src_attr, secondary=join_table,
                primaryjoin=Identity.id == join_table.c.src,
                secondaryjoin=Identity.id == join_table.c.dst,
                cascade_backrefs=False, **kwargs,
            ),
            src_attr: relationship(
                src_model, back_populates=dst_attr, secondary=join_table,
                primaryjoin=Identity.id == join_table.c.dst,
                secondaryjoin=Identity.id == join_table.c.src,
                cascade_backrefs=False, **kwargs,
            ),
        }
