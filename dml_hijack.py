import logging as logger
import torch
from modules.dml import directml_init, directml_do_hijack
directml_init()
directml_do_hijack()
for d in range(torch.cuda.device_count()):
    logger.debug(f"디바이스 {d}: {torch.cuda.get_device_name(d)}")
device = "0"
