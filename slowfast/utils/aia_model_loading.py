"""AIA to PyTorch checkpoint name converting utility."""

import re


def get_name_convert_func():
    """
    Get the function to convert AIA layer names to SlowFast layer names.
    Returns:
        (func): function to convert parameter name from AIA format to PyTorch
        format.
    """

    pairs = [
        # fuse fast to slow
        # -----------------------------------------------------
        # fast.Tconv1.conv.weight -> s1_fuse.conv_f2s.weight
        [r"^fast.Tconv([1-4]).conv.(.*)", r"s\1_fuse.conv_f2s.\2"],

        # pathway
        # -----------------------------------------------------
        # slow -> pathway0, fast -> pathway1
        [r"^slow(.*)", r"pathway0_\1"],
        [r"^fast(.*)", r"pathway1_\1"],

        # stem
        # ----------------------------------------------------
        # slow.conv1.weight -> s1.pathway0_stem.conv.weight
        [r"(.*).conv1.weight", r"s0.\1stem.conv.weight"],
        # slow.bn1.weight -> s1.pathway0_stem.bn.weight
        [r"(.*).bn1(.*)", r"s0.\1stem.bn\2"],

        # res stage
        # -----------------------------------------------------
        # conv1 -> a
        [r"(.*).conv1.(.*)", r"\1.a.\2",],
        # conv2 -> b
        [r"(.*).conv2.(.*)", r"\1.b.\2",],
        # conv3 -> c
        [r"(.*).conv3.(.*)", r"\1.c.\2",],
        # btnk -> branch2
        [r"(.*).btnk.(.*)", r"\1.branch2.\2",],
        # shortcut -> branch1
        [r"(.*).shortcut.(.*)", r"\1.branch1.\2",],
        # conv.weight -> weight
        [r"(.*)([abc123]).conv.weight\Z", r"\1\2.weight"],
        # .bn. -> _bn.
        [r"(.*)([abc123]).bn\.(.*)", r"\1\2_bn.\3"],

        # res_nl1 -> s1
        [r"(.*).res_nl([1-4])(.*)", r"s\2.\1\3"],
        # .res_0 -> _res0
        [r"(.*).res_([0-9]+)(.*)", r"\1res\2\3"],

        # stage number
        [r"^s4\.(.*)", r"s5.\1"],
        [r"^s3\.(.*)", r"s4.\1"],
        [r"^s2\.(.*)", r"s3.\1"],
        [r"^s1\.(.*)", r"s2.\1"],
        [r"^s0\.(.*)", r"s1.\1"],

        # head
        # -----------------------------------------------------
        # cls_head.pred.weight -> head.projection.weight
        [r"cls_head.pred", r"head.projection"],
    ]
    
    def convert_aia_name_to_pytorch(aia_layer_name):
        """
        Convert the aia_layer_name to slowfast format by apply the list of
        regular expressions.
        Args:
            aia_layer_name (str): aia layer name.
        Returns:
            (str): pytorch layer name.
        """
        if aia_layer_name.startswith("module"):
            aia_layer_name = aia_layer_name.split("module.")[1]
        if aia_layer_name.startswith("backbone"):
            aia_layer_name = aia_layer_name.split("backbone.")[1]
        for source, dest in pairs:
            aia_layer_name = re.sub(source, dest, aia_layer_name)
        return aia_layer_name

    return convert_aia_name_to_pytorch
