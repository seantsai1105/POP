# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
try:
    from model.vlm_modules.internvl3.internvl.model.internvl_chat.configuration_intern_vit import InternVisionConfig
    from model.vlm_modules.internvl3.internvl.model.internvl_chat.configuration_internvl_chat import InternVLChatConfig
    from model.vlm_modules.internvl3.internvl.model.internvl_chat.modeling_intern_vit import InternVisionModel
    from model.vlm_modules.internvl3.internvl.model.internvl_chat.modeling_internvl_chat import InternVLChatModel
except:
    from .configuration_intern_vit import InternVisionConfig
    from .configuration_internvl_chat import InternVLChatConfig
    from .modeling_intern_vit import InternVisionModel
    from .modeling_internvl_chat import InternVLChatModel

__all__ = ['InternVisionConfig', 'InternVisionModel',
           'InternVLChatConfig', 'InternVLChatModel']
