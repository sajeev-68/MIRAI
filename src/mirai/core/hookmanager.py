from mirai.core.types import LayerRecord
import torch
import torch.nn as nn


class HookManager:
    def __init__(self, model: nn.Module, x: torch.Tensor):
        """
        Initialize hook manager.
        :param model: Frozen model
        :param x: input tensor to forward through the model
        """
        self.model = model
        self.inp = x
        self._pre_activations = []   # list of (layer_name, tensor)
        self._post_activations = []  # list of (layer_name, tensor)
        self.lr = {}                 # dict[layer_name -> LayerRecord]

    def make_pre_hook(self, layer_name: str):
        """
        Returns a pre-hook callback for a given layer.
        Captures z = Wx + b (before activation).
        """
        def pre_hook(module, layer_input):
            # layer_input is a tuple — take first element
            tensor = layer_input[0].detach().cpu().clone()
            self._pre_activations.append((layer_name, tensor))
        return pre_hook

    def make_post_hook(self, layer_name: str):
        """
        Returns a forward hook callback for a given layer.
        Captures a = activation(z) (after activation).
        """
        def post_hook(module, layer_input, layer_output):
            tensor = layer_output.detach().cpu().clone()
            self._post_activations.append((layer_name, tensor))
        return post_hook

    def _hook_layer(self) -> dict[str, LayerRecord]:
        """
        Walk every layer in the model, attach hooks, run one forward pass,
        detach hooks, then build and return a dict of LayerRecords.
        """
        hook_handles = []

        # named_modules() walks all layers at every depth
        # skip the top-level model itself (name == "")
        for name, layer in self.model.named_modules():
            if name == "":
                continue
            h_pre = layer.register_forward_pre_hook(self.make_pre_hook(name))
            h_post = layer.register_forward_hook(self.make_post_hook(name))
            hook_handles.extend([h_pre, h_post])

        # single forward pass — hooks fire automatically in order
        with torch.no_grad():
            self.model(self.inp)

        # detach all hooks immediately — model is clean after this
        for h in hook_handles:
            h.remove()

        # build a lookup for fast weight/bias access
        module_lookup = {name: layer for name, layer in self.model.named_modules()}

        # zip pre and post activations — they are in the same order
        for (name, tensor_pre), (_, tensor_post) in zip(
            self._pre_activations, self._post_activations
        ):
            relu_mask = (tensor_post > 0)

            layer = module_lookup.get(name)
            weight = layer.weight.detach().cpu().clone() if layer is not None and hasattr(layer, 'weight') and layer.weight is not None else None
            bias   = layer.bias.detach().cpu().clone()   if layer is not None and hasattr(layer, 'bias')   and layer.bias   is not None else None

            self.lr[name] = LayerRecord(
                pre_activation=tensor_pre,
                post_activation=tensor_post,
                relu_mask=relu_mask,
                weight=weight,
                bias=bias,
            )

        return self.lr