"""P2O entrypoint: native train_async plus a drained GEPA phase adapter."""

import asyncio
from functools import partial

from examples.seta.p2o.driver import PhaseHandler, prepare_resume, select_paired_checkpoint
from train_async import train

from miles.utils.arguments import parse_args
from miles.utils.async_utils import with_disposer

if __name__ == "__main__":
    args = parse_args(preprocess_args=select_paired_checkpoint)
    prepare_resume(args)
    asyncio.run(with_disposer(partial(train, phase_factory=PhaseHandler), args))
