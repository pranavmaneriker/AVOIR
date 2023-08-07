import functools
from typing import Callable, Optional
import logging
import copy
import math

from .grammar import *
from .grammar.specification import Specification
from .grammar.errors import NoObservedOutcomesError

logger = logging.getLogger("dsl")
logger.setLevel(level=logging.CRITICAL)


class SpecTracker:
    """
    Used to keep track of all specifications
    Allows only one spec per function which is initialized at the time of the first call
    Modifying the method_specs object here will modify the tracked function
    """
    method_specs: Dict[Callable, Specification] = {}
    call_count: Dict[Callable, int] = {}

    @staticmethod
    def called(f: Callable, sp: Specification):
        if SpecTracker.get_spec(f) is None:
            SpecTracker.method_specs[f] = sp
            SpecTracker.call_count[f] = 1

    @staticmethod
    def get_spec(f: Callable):
        return SpecTracker.method_specs.get(f, None)

    @staticmethod
    def get_call_count(f: Callable):
        return SpecTracker.call_count.get(f, 0)

    @staticmethod
    def update_count(f: Callable):
        SpecTracker.call_count[f] += 1


class ObservedFunction:
    def __init__(self, func, spec: Specification, observation_key: Optional[str] = None, with_bounds=False, delta=0.05,
                 optimization_frequency=10, our_approach=True):
        self.func = func
        self.observation_key = observation_key
        self.with_bounds = with_bounds
        self.saved_spec = spec
        self.delta = delta
        self.optimization_frequency = optimization_frequency
        self.with_bounds = with_bounds
        self.our_approach = our_approach
        self.optimization_achieved = False
        self.optimization_achieved_step = math.inf
        self.num_terms = 1
        if with_bounds:
            spec.prepare_for_opt(is_top_level=True)
            self.num_terms = spec.num_E
        self.spec_val_history = []
        self.best_achieved = math.inf
        #if with_bounds:
           #spec.prepare_for_opt(is_top_level=True)

    @property
    def spec(self) -> Specification:
        return SpecTracker.get_spec(self.func)

    def undo(self, key: str):
        my_spec = SpecTracker.get_spec(self.func)
        my_count = SpecTracker.get_call_count(self.func)
        my_spec.unobserve(call_id=my_count, observation_key=key)

    def __call__(self, *args, **kwargs):
        if len(args) > 0:
            raise NotImplementedError("Specs only implemented for functions with keyword args")
        r = self.func(**kwargs)  # evaluate the function
        #if self.with_bounds:
        #    my_spec.eval_bounded_at_delta(self.delta, my_count)
        bindings = dict(kwargs)
        bindings[NumericalExpression.RESERVE_WORD_RETVAL] = r
        my_spec = SpecTracker.get_spec(self.func)
        my_count = SpecTracker.get_call_count(self.func)
        key = kwargs[self.observation_key] if self.observation_key in kwargs else None
        SpecTracker.update_count(self.func)
        my_spec.observe(bindings, call_id=my_count, observation_key=key)
        try:
            if self.with_bounds:
                if my_count % self.optimization_frequency == 0 and my_count > 0:
                    #ipdb.set_trace()
                    if self.our_approach:
                        if not self.optimization_achieved:
                            logger.info("Running optimizer")
                            eval_spec = my_spec.eval_bounded_with_constraints_at_delta(delta=self.delta, call_id=my_count)
                            delta_bindings = my_spec.get_delta_bindings()
                            my_spec.eval_bounded_at_delta(delta_bindings)
                            self.best_achieved = min(self.best_achieved, eval_spec.bound_delta)
                            if eval_spec.bound_delta <= self.delta:
                                logger.info("Optimality achieved")
                                self.optimization_achieved = True
                                self.optimization_achieved_step = my_count
                                self.delta = delta_bindings
                            else:
                                my_spec.eval_bounded_at_delta(self.delta)
                        else:
                            logger.info("Running with optimized values")
                            eval_spec = my_spec.eval_bounded_at_delta(delta=self.delta)
                    else:
                        logger.info("Running OOPSLA method")
                        delta = self.delta/self.num_terms
                        eval_spec = my_spec.eval_bounded_at_delta(delta=delta)
                        if not self.optimization_achieved and eval_spec.bound_delta <= self.delta:
                            self.optimization_achieved = True
                            self.optimization_achieved_step = my_count

                    self.spec_val_history.append((my_count, copy.deepcopy(eval_spec.bounded_observations)))
        except NoObservedOutcomesError:
            logging.info("Could not optimize because insufficient values to evaluate at least one subexpression")
        except ZeroDivisionError:
            logging.info("Could not optimize because the Expected value of a term in the denominator was 0")
        except ValueError as e:
            if str(e).startswith("Cannot load a SolverResults object with bad status"):
                logging.info("Error during solver run")
            else:
                raise e
        if self.with_bounds:
            return r, self.spec_val_history
        else:
            return r


def spec(sp: Specification, observation_key: Optional[str] = None, include_confidence=False, delta=0.05,
         optimization_frequency=10, our_approach=True):
    """
    :param sp: The specification that the decorated function will be monitored
    with respect to.
    :param observation_key: The key from the kwargs of the decorated function
    that will be used to identify unique observations. If an observation key is
    provided that was previously seen is observed, then the previously seen
    observation is updated as opposed to adding a new observation.
    """
    def decorator_spec(func):
        observed_function = ObservedFunction(func, sp, observation_key,
                                             with_bounds=include_confidence,
                                             delta=delta,
                                             optimization_frequency=optimization_frequency,
                                             our_approach=our_approach)
        observed_function = functools.wraps(func)(observed_function)
        SpecTracker.called(observed_function, sp)  # update the tracker with the spec for this function
        SpecTracker.called(func, sp)  # This one dirty trick can help you solve the issue
        # of correct bindings without figuring out decorators
        return observed_function

    return decorator_spec
