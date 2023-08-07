from typing import Callable
import pandas as pd
from ...grammar.specification import Specification
from ...dsl import spec


"""
The Database class serves as a simulation of a database that supports
materialized views. It does this with the use of pandas DataFrames
"""


class MonitoredView:
    def __init__(self, query: Callable, data: pd.DataFrame,
                 sp: Specification, observation_key: str, spec_args=None, progress_bar=None):
        self.query = query
        # observer set in `update_spec(sp)`
        self.observer = None
        self.key = observation_key
        self.cache = query(data)
        self.spec = self.update_spec(sp, spec_args)
        self.refresh(data, progress_bar)

    def update_spec(self, sp: Specification, spec_args):
        if spec_args is None:
            spec_args = {}
        if not isinstance(sp, Specification):
            raise ValueError("Input provided was not parsed into a specification")

        @spec(sp, observation_key=self.key, **spec_args)
        def observe(**attributes):
            return
        self.observer = observe

    def refresh(self, data: pd.DataFrame, progress_bar=None):
        new_view = self.query(data)
        # delete old records
        stale_record_ids = self.cache[self.key][~self.cache[self.key].isin(
            new_view[self.key])]
        for stale_id in stale_record_ids:
            self.observer.undo(stale_id)
        # add new and update records
        iterator = new_view.iterrows()
        if progress_bar:
            iterator = progress_bar(new_view.iterrows())

        for _, row in iterator:
            self.observer(**row.to_dict())
        self.cache = new_view

    def status(self):
        return self.observer.spec.eval()
        #return self.observer.spec.eval_bounded()

    def status_bounded(self, delta=0.05):
        return self.observer.spec.eval_bounded_at_delta(delta)


class ViewNotFoundError(Exception):
    pass


class Database:
    def __init__(self, initial_data: pd.DataFrame):
        self.data = initial_data
        self.views = {}

    def create_materialized_view(self, view_name: str, query: Callable,
                                 specification: Specification, observation_key: str,
                                 spec_args=None, progress_bar=None):
        self.views[view_name] = MonitoredView(
            query,
            self.data,
            specification,
            observation_key,
            spec_args,
            progress_bar
        )

    def refresh_materialized_view(self, view_name):
        if view_name not in self.views:
            raise ViewNotFoundError(
                f"No view named {view_name} found in database")

        self.views[view_name].refresh(self.data)
