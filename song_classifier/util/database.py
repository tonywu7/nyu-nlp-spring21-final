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
import re
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import date, datetime, timezone
from functools import wraps
from operator import attrgetter, itemgetter
from sqlite3 import Connection as SQLite3Connection
from typing import (Any, Callable, Dict, List, Optional, Set, Tuple, Type,
                    TypeVar, Union, overload)
from urllib.parse import urlencode

import simplejson as json
import udatetime
import unidecode
from sqlalchemy import MetaData, Table, create_engine, event, types
from sqlalchemy.engine import Connection, Engine, Result
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.associationproxy import (AssociationProxy,
                                             AssociationProxyInstance)
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import (Mapper, Query, RelationshipProperty, Session,
                            aliased, declarative_base, declared_attr,
                            relationship, scoped_session, sessionmaker,
                            with_polymorphic)
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.orm.properties import ColumnProperty
from sqlalchemy.schema import (DDL, Column, ForeignKey, Index,
                               PrimaryKeyConstraint, UniqueConstraint)
from sqlalchemy.sql.expression import FunctionElement, column, select, table
from sqlalchemy.sql.functions import count
from sqlalchemy.sql.selectable import \
    LABEL_STYLE_TABLENAME_PLUS_COL as LS_TABLE_COL
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

    __reflection__: Reflection

    @classmethod
    def _init_reflection(cls):
        cls.__reflection__ = Reflection(cls)


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


class FTS5:
    def __init__(self):
        self.ident = Identity.__reflection__
        self.polymorph = with_polymorphic(Identity, '*')
        self.sessionmaker: Callable[[], Session]

        self.selectable = Query(self.polymorph).statement.set_label_style(LS_TABLE_COL)
        self.columns = self.indexed_columns()
        self.rowid_c = self.translated(self.ident.mapper.c.id)
        self.model_c = self.translated(self.ident.mapper.c.model)
        self.idx_t = self.idx_table(self.ident.mapper)
        self.idx_p = aliased(Identity, self.idx_t, adapt_on_names=True)

    @property
    def session(self) -> Session:
        return self.sessionmaker()

    @property
    def initialized(self) -> bool:
        return hasattr(self, 'sessionmaker')

    @property
    def view_name(self):
        return 'identity_view'

    @property
    def idx_name(self):
        return 'identity_idx'

    def indexed_columns(self) -> List[Column]:
        return [c for c in self.selectable.subquery().c]

    def translated(self, target: Column) -> Column:
        for col in self.selectable.subquery().c:
            if col.base_columns == target.base_columns:
                return col

    def idx_table(self, mapper: Mapper) -> Table:
        columns = []
        for c in mapper.columns:
            translated = self.translated(c)
            args = [translated.key, c.type]
            if translated.foreign_keys:
                for foreign_key in translated.foreign_keys:
                    args.append(ForeignKey(foreign_key.column))
            columns.append(Column(*args, key=c.key, primary_key=c.primary_key))
        return Table(
            self.idx_name, metadata,
            Column('identity_idx', types.String(), key='master'),
            *columns,
            keep_existing=True,
        )

    def polymorphic_view(self) -> DDL:
        template = """
        CREATE VIEW IF NOT EXISTS %(name)s
        AS %(select)s
        """
        info = {
            'name': self.view_name,
            'select': self.selectable.compile(),
        }
        return DDL(template % info)

    def fts_virtual_table(self) -> DDL:
        template = """
        CREATE VIRTUAL TABLE IF NOT EXISTS %(name)s
        USING fts5(%(columns)s, content=%(view_name)s, content_rowid=%(rowid_name)s)
        """
        info = {
            'name': self.idx_name,
            'columns': ', '.join([c.key for c in self.columns]),
            'view_name': self.view_name,
            'rowid_name': self.rowid_c.key,
        }
        return DDL(template % info)

    def init(self, sessionmaker: Callable[[], Session]):
        self.sessionmaker = sessionmaker
        session = self.session
        view = self.polymorphic_view()
        fts = self.fts_virtual_table()
        view.execute(session.bind)
        fts.execute(session.bind)
        event.listen(session, 'before_flush', self.preflush_delete)
        event.listen(session, 'after_flush', self.postflush_update)
        session.commit()

    def preflush_delete(self, session: Session, context, instances):
        ids = [str(item.id) for item in [*session.dirty, *session.deleted]]
        stmt = """
        INSERT INTO %(name)s(%(name)s, rowid, %(columns)s)
        SELECT 'delete', %(rowid_name)s, * FROM %(view_name)s
        WHERE %(rowid_name)s IN (%(ids)s)
        """
        info = {
            'name': self.idx_name,
            'columns': ', '.join([c.key for c in self.columns]),
            'view_name': self.view_name,
            'rowid_name': self.rowid_c.key,
            'ids': ', '.join(ids),
        }
        session.execute(stmt % info)

    def postflush_update(self, session: Session, context):
        ids = [str(item.id) for item in [*session.new, *session.dirty]]
        stmt = """
        INSERT INTO %(name)s(rowid, %(columns)s)
        SELECT %(rowid_name)s, * FROM %(view_name)s
        WHERE %(rowid_name)s IN (%(ids)s)
        """
        info = {
            'name': self.idx_name,
            'columns': ', '.join([c.key for c in self.columns]),
            'view_name': self.view_name,
            'rowid_name': self.rowid_c.key,
            'ids': ', '.join(ids),
        }
        session.execute(stmt % info)

    def destroy(self, session: Optional[Session] = None):
        session = session or self.session
        session.execute(f'DROP TABLE IF EXISTS {self.idx_name}')
        session.execute(f'DROP VIEW IF EXISTS {self.view_name}')
        session.commit()

    def rebuild(self):
        session = self.session
        session.execute(f"INSERT INTO {self.idx_name}({self.idx_name}) VALUES('rebuild');")
        session.commit()

    def query(self, q: Optional[str] = None) -> Query:
        clause = self.idx_t.c.id.isnot(None)
        if q is not None:
            clause = clause & self.idx_t.c.master.op('match')(q)
        return self.session.query(self.idx_p).filter(clause)

    def tokenized(self, q: Optional[str] = None) -> str:
        if q is None:
            return None
        return slugify(q, sep='* ') + '*'

    def search(self, q: Optional[str] = None) -> Query:
        return self.query(self.tokenized(q))

    def instanceof(self, model: Type[T], q: Optional[str] = None) -> Query:
        desc = [m.entity.__name__ for m in model.__mapper__.self_and_descendants]
        targets = ' OR '.join([f'{self.model_c.key}:{d}' for d in desc])
        if q is not None:
            query = f'({targets}) AND {slugify(q, sep="* ")}*'
        else:
            query = f'({targets})'
        return self.query(query)

    @contextmanager
    def using_mapper(self, model: Type[T]):
        try:
            metadata.remove(self.idx_t)
            self.idx_t = self.idx_table(inspect(model))
            self.idx_p = aliased(model, self.idx_t, adapt_on_names=True)
            yield self.idx_p
        finally:
            metadata.remove(self.idx_t)
            self.idx_t = self.idx_table(inspect(Identity))
            self.idx_p = aliased(Identity, self.idx_t, adapt_on_names=True)

    def ids(self, q: Optional[str] = None, raw_query=False) -> Query:
        if not raw_query:
            q = self.tokenized(q)
        return self.session.query(self.idx_t.c.id).filter(self.idx_t.c.master.op('match')(q))

    def all(self, model: Type[T], q: Optional[str] = None) -> List[T]:
        return self.instanceof(model, q).all()

    def lookup(self, model: Type[T], q: Optional[str] = None) -> Query:
        return self.session.query(model).filter(model.id.in_(self.ids(q)))


def slugify(name: str, sep='-', *, limit=0) -> str:
    t = re.sub(r'[\W_]+', sep, str(unidecode.unidecode(name))).strip(sep).lower()
    if limit > 0:
        t = sep.join(t.split(sep)[:limit])
    return t


class Reflection:
    mapper: Mapper
    local_table: Table
    mapped_table: Table

    attributes: Dict[str, InferrableSelectable]

    relationships: Dict[str, RelationshipProperty]
    proxies: Dict[str, AssociationProxy]

    atypes: Dict[str, Type]
    ctypes: Dict[str, Type]
    dtypes: Dict[str, Type]

    autoincrement: Tuple[str, ...]
    primary_key: Tuple[str, ...]
    unique_columns: Tuple[Tuple[str, ...], ...]

    polymorphic_on: Optional[Column]
    polymorphic_ident: Optional[Any]

    ancestral_columns: List[Column]
    ancestral_identity: List[Column]

    lineage: List[Mapper]

    @property
    def columns(self) -> Dict[str, Column]:
        return {c.name: c for c in self.mapper.columns}

    def get_unique_attrs(self, obj):
        values = []
        for cols in self.unique_columns:
            values.append(attrgetter(*cols)(obj))
        return tuple(values)

    def get_unique_items(self, info):
        values = []
        for cols in self.unique_columns:
            values.append(itemgetter(*cols)(info))
        return tuple(values)

    @staticmethod
    def _find_unique_columns(table: Table) -> Tuple[Set[Tuple[Tuple[str, ...], ...]], Optional[Tuple[str, ...]]]:
        unique = set()
        primary_key = None
        for c in table.constraints:
            cols = tuple(sorted(c.name for c in c.columns))
            if isinstance(c, PrimaryKeyConstraint):
                primary_key = cols
                unique.add(cols)
            if isinstance(c, UniqueConstraint):
                unique.add(cols)
        for i in table.indexes:
            if i.unique:
                cols = tuple(sorted(c.name for c in i.columns))
                unique.add(cols)
        return unique, primary_key

    @staticmethod
    def _find_lineage(mapper: Mapper) -> List[Mapper]:
        line = []
        while mapper is not None:
            line.append(mapper)
            mapper = mapper.inherits
        return line

    def __init__(self, model: Type):
        table: Table = model.__table__
        mapper: Mapper = inspect(model)

        self.mapper = mapper
        self.local_table = mapper.local_table
        self.mapped_table = mapper.persist_selectable

        self.ancestral_columns = []
        self.ancestral_identity = []
        self.lineage = []

        self.polymorphic_on = None
        self.polymorphic_ident = None

        unique, primary_key = self._find_unique_columns(table)
        self.primary_key = primary_key
        self.unique_columns = tuple(sorted(unique))

        self.polymorphic_on = mapper.polymorphic_on
        self.polymorphic_ident = mapper.polymorphic_identity

        self.lineage = self._find_lineage(mapper)
        for t in mapper.tables:
            if t is table:
                continue
            self.ancestral_columns.extend(t.columns)
            self.ancestral_identity.extend([c for c in t.columns if c.primary_key])

        self.relationships = {r.key: r for r in mapper.relationships}
        proxies = {r.info.get('key'): r for r in mapper.all_orm_descriptors
                   if isinstance(r, AssociationProxy)}
        self.proxies = {k: v.for_class(model) for k, v in proxies.items() if k}

        target_attr_types = (InstrumentedAttribute, AssociationProxy)
        self.attributes = {**{c.key: c for c in self.mapper.all_orm_descriptors
                              if isinstance(c, target_attr_types) and c.key[0] != '_'},
                           **self.proxies}

        self.atypes = atypes = {}
        self.ctypes = ctypes = {}
        self.dtypes = dtypes = {}

        cls_annotations = getattr(model, '__annotations__')
        for c, col in self.columns.items():
            atypes[c] = ColumnProperty
            ctypes[c] = col.type
            if cls_annotations:
                dt = cls_annotations.get(c)
                if isinstance(dt, type):
                    dtypes[c] = dt
                    continue
            dtypes[c] = object

        for c, r in self.relationships.items():
            atypes[c] = RelationshipProperty
            dtypes[c] = self._detect_collection(r.collection_class)

        for c, p in self.proxies.items():
            atypes[c] = AssociationProxyInstance
            dtypes[c] = self._detect_collection(p.local_attr.property.collection_class)

    @classmethod
    def _detect_collection(cls, type_):
        type_ = type_ or list
        try:
            return type(type_())
        except Exception:
            return type_

    @classmethod
    def is_column(cls, attr: InferrableSelectable):
        if isinstance(attr, InstrumentedAttribute):
            attr = attr.prop
        return isinstance(attr, ColumnProperty)

    @classmethod
    def is_relationship(cls, attr: InferrableSelectable):
        if isinstance(attr, InstrumentedAttribute):
            attr = attr.prop
        return isinstance(attr, RelationshipProperty)

    @classmethod
    def is_proxy(cls, attr: InferrableSelectable):
        return isinstance(attr, (AssociationProxy, AssociationProxyInstance))

    @classmethod
    def owning_class(cls, attr: InferrableSelectable):
        if cls.is_column(attr) or cls.is_relationship(attr):
            return attr.parent.entity
        if cls.is_proxy(attr):
            return attr.owning_class

    @classmethod
    def join_target(cls, attr: InferrableSelectable) -> Mapper | None:
        if cls.is_relationship(attr):
            return attr.property.entity
        if cls.is_proxy(attr):
            remote_prop = attr.remote_attr.property
            if isinstance(remote_prop, RelationshipProperty):
                return remote_prop.entity


@event.listens_for(metadata, 'after_create')
def find_models(*args, **kwargs):
    for k, v in Base.registry._class_registry.items():
        if isinstance(v, type) and issubclass(v, Identity):
            v._init_reflection()
