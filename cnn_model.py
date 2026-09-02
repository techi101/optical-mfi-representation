"""
cnn_model.py
============
STEP 2: define the neural networks.

This file defines THREE networks, one for each way of representing the same
received optical signal. That is the core of the paper's argument, so it is
worth stating clearly:

    representation            network            pre-processing done by a human
    ------------------------  -----------------  ------------------------------
    64x64 constellation image ConstellationCNN   binning into a 2-D histogram
    512 raw I/Q samples       IQCNN (1-D)        none at all
    15 statistical features   FeatureMLP         a lot (chosen cumulants etc.)

All three see the SAME physical signals. They differ only in how those
signals are presented. Comparing them answers a question that a plain
"CNN vs SVM" comparison cannot:

    Is the benefit coming from the DEEP MODEL, or from the REPRESENTATION?

If the FeatureMLP (a deep model on hand-crafted features) performs like the
SVM rather than like the CNN, then the representation is what matters -- and
that is a much more interesting and defensible claim than "neural networks
are better".

WHY A CNN IS THE RIGHT TOOL FOR THE IMAGE
A constellation diagram is a picture, and what identifies the format is a
spatial pattern: "4 blobs, or a 4x4 grid, or an 8x8 grid, or a cross?".
A convolution slides a small 3x3 filter across the whole image, reusing the
SAME weights everywhere, so a "bright blob" detector learned once works at
every position. That weight sharing is why a CNN needs far fewer parameters
than a fully-connected network on images: a dense network would have to learn
what a blob looks like separately at all 4096 pixel positions.
"""

import numpy as np
import torch
import torch.nn as nn

import config


# ===========================================================================
# MODEL 1 -- THE 2-D CNN ON CONSTELLATION IMAGES
# ===========================================================================
class ConstellationCNN(nn.Module):
    """
    A deliberately small CNN: `n_blocks` convolution blocks then 2 dense layers.

    DATA SHAPE THROUGH THE DEFAULT NETWORK (channels x height x width)

        input                     1 x 64 x 64      the constellation image
        block 1  conv 1->16        16 x 64 x 64
                 maxpool 2         16 x 32 x 32
        block 2  conv 16->32       32 x 32 x 32
                 maxpool 2         32 x 16 x 16
        block 3  conv 32->64       64 x 16 x 16
                 maxpool 2         64 x  8 x  8
        flatten                    4096
        dropout + fc 4096->64        64
        fc 64->5                      5          one score per class

    WHY 3 CONVOLUTION BLOCKS AND NOT 10?
        Each block halves the image. After 3 blocks one pixel of the feature
        map corresponds to roughly a 30x30 patch of the original 64x64 image
        -- big enough to contain several constellation clusters, which is
        exactly the scale of the pattern we need to recognise. More blocks
        would shrink the map below the size of the pattern and add parameters
        with nothing left to learn. Our images are simple (no texture, no
        colour, no clutter), so depth is not needed. Deep networks like
        ResNet-50 exist for photographs of the real world.
        The ablation study in experiments.py TESTS this claim rather than
        just asserting it.

    WHY 16 -> 32 -> 64 CHANNELS?
        The standard pyramid: as the image gets smaller, allow more feature
        types. Early on there are few useful patterns (blobs, edges); later
        there are many (grid arrangements, cross shapes). Doubling each time
        is the conventional choice.

    WHY 3x3 FILTERS?
        3x3 is the smallest filter with a notion of "centre and surround",
        i.e. direction. Two stacked 3x3 layers see the same area as one 5x5
        but with fewer parameters and an extra non-linearity in between. The
        default since VGG (2014).
    """

    def __init__(self, n_classes=None, dropout=0.3, n_blocks=3, base_ch=16,
                 img_size=None):
        super().__init__()
        n_classes = config.N_CLASSES if n_classes is None else n_classes
        img_size = config.IMG_SIZE if img_size is None else img_size

        layers = []
        in_ch = 1
        ch = base_ch
        for _ in range(n_blocks):
            layers.append(self._block(in_ch, ch))
            in_ch = ch
            ch *= 2
        self.features = nn.Sequential(*layers)

        # Work out the flattened size by actually pushing a dummy tensor
        # through, rather than deriving it by hand. Safer when n_blocks or
        # img_size change.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, img_size, img_size)
            flat = self.features(dummy).numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Dropout randomly zeroes 30% of these numbers during training
            # (and does nothing at test time). It stops the network relying on
            # any single feature and is our main defence against overfitting.
            nn.Dropout(dropout),
            nn.Linear(flat, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_classes),        # raw scores ("logits")
        )
        # NOTE: no Softmax at the end. PyTorch's CrossEntropyLoss applies it
        # internally in a numerically safer way. For probabilities, apply
        # torch.softmax to the output yourself.

    @staticmethod
    def _block(in_ch, out_ch):
        return nn.Sequential(
            # padding=1 with a 3x3 kernel keeps width/height unchanged, so
            # only the pooling layer changes the size. Easier to reason about.
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            # BatchNorm rescales layer outputs to roughly zero mean and unit
            # variance, keeping the numbers in a healthy range. That makes
            # training faster and much less sensitive to the learning rate.
            # bias=False above because BatchNorm has its own shift term.
            nn.BatchNorm2d(out_ch),
            # ReLU = max(0, x). Without a non-linearity, stacked layers would
            # collapse into a single linear layer.
            nn.ReLU(inplace=True),
            # MaxPool keeps the largest value in each 2x2 square, halving the
            # image. It preserves "was this pattern present?" and discards
            # "exactly which pixel", so a cluster shifted by one pixel is
            # still the same cluster.
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        """x: (batch, 1, H, W) float tensor scaled to roughly 0..1"""
        return self.classifier(self.features(x))


# ===========================================================================
# WHY WE FLATTEN INSTEAD OF GLOBAL-AVERAGE-POOLING (image model only)
# ===========================================================================
# Many modern CNNs replace Flatten with a global average pool, which discards
# WHERE each feature was found. For photographs that is good -- a cat is a cat
# wherever it appears. Here it would be harmful: the whole point is the
# geometric ARRANGEMENT of clusters in the I-Q plane. So we flatten and keep
# the positions. Note the 1-D model below makes the OPPOSITE choice, for a
# reason explained there.


# ===========================================================================
# MODEL 2 -- THE 1-D CNN ON RAW I/Q SAMPLES
# ===========================================================================
class IQCNN(nn.Module):
    """
    A 1-D CNN that reads the raw I/Q sample stream directly: no binning, no
    hand-crafted features, no human pre-processing at all.

    INPUT SHAPE: (batch, 2, 512)
        channel 0 = the in-phase (I) samples
        channel 1 = the quadrature (Q) samples
        512 consecutive received symbols

    WHY THIS MODEL IS IN THE PAPER
    It is the "zero pre-processing" end of the scale. If it matched the image
    CNN, we would conclude that pre-processing does not matter and any deep
    model works. If it is worse, we learn something more interesting: that
    turning the samples into a 2-D density picture is doing real work that the
    network cannot easily do for itself.

    WHY WE EXPECT IT TO STRUGGLE (state this prediction in the paper, then
    check it against the result)
    The transmitted symbols are independent and identically distributed. That
    means the ORDER of the samples carries almost no information -- shuffling
    them would barely change the answer. But a 1-D convolution is built
    precisely to exploit order. So the model has to work out, from a sequence,
    the shape of the DISTRIBUTION those samples were drawn from. The
    constellation image hands that distribution over directly: a 2-D histogram
    IS the estimated distribution. This is the clearest statement of why
    representation matters here.

    WHY GLOBAL AVERAGE POOLING HERE, WHEN THE IMAGE MODEL USES FLATTEN?
    Because the signal is stationary: symbol number 7 is statistically
    identical to symbol number 400, so absolute position in the sequence is
    meaningless and averaging over it is exactly right. In the image the
    opposite holds -- position in the I-Q plane is the entire signal. Making
    opposite choices for opposite reasons is a good thing to be able to
    explain in a viva.

    WHY LARGER KERNELS (7, then 5, then 3)?
    A wide first kernel lets the first layer see a short run of symbols at
    once, which is what is needed to pick up correlations such as the shared
    phase error within a carrier-recovery block.
    """

    def __init__(self, n_classes=None, n_samples=None, dropout=0.3):
        super().__init__()
        n_classes = config.N_CLASSES if n_classes is None else n_classes
        n_samples = config.N_IQ_SYMBOLS if n_samples is None else n_samples

        def block(in_ch, out_ch, k):
            return nn.Sequential(
                nn.Conv1d(in_ch, out_ch, k, padding=k // 2, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
            )

        self.features = nn.Sequential(
            block(2, 32, 7),        # (2,512)  -> (32,256)
            block(32, 64, 5),       #          -> (64,128)
            block(64, 64, 3),       #          -> (64, 64)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)      # -> (64, 1), see docstring
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        """x: (batch, 2, n_samples) float tensor"""
        return self.classifier(self.pool(self.features(x)))


# ===========================================================================
# MODEL 3 -- THE MLP ON HAND-CRAFTED FEATURES
# ===========================================================================
class FeatureMLP(nn.Module):
    """
    A plain fully-connected network on the same 15 statistical features that
    the SVM and k-NN receive.

    WHY THIS MODEL IS THE MOST IMPORTANT BASELINE IN THE PAPER
    Without it, "the CNN beats the SVM" is ambiguous, because TWO things
    changed at once:
        (a) the model became deep and learned, and
        (b) the input became a full 2-D image instead of 15 numbers.
    A reviewer will point this out immediately. The FeatureMLP holds (a)
    fixed and changes only (b): it is a deep, trained neural network, but it
    sees only the 15 features. So:

        if FeatureMLP ~ SVM  -> the REPRESENTATION is what matters
        if FeatureMLP ~ CNN  -> the DEEP MODEL is what matters

    Either outcome is a real finding. Running this experiment is the
    difference between a project report and a paper.

    ARCHITECTURE: 15 -> 128 -> 64 -> n_classes.
    Deliberately given MORE capacity than it needs (about 10k parameters for
    15 inputs) so that nobody can argue it lost because it was too small.
    """

    def __init__(self, n_classes=None, n_features=15, hidden=(128, 64),
                 dropout=0.2):
        super().__init__()
        n_classes = config.N_CLASSES if n_classes is None else n_classes

        layers = []
        prev = n_features
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """x: (batch, n_features) float tensor, already standardised"""
        return self.net(x)


# ===========================================================================
# HELPERS
# ===========================================================================
def count_parameters(model):
    """How many numbers the network has to learn."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model, input_shape):
    """
    Rough multiply-accumulate count for ONE forward pass, in MACs.

    Why this matters for the paper: an MFI block would eventually live inside
    a receiver's FPGA or DSP, where compute budget is tight. Accuracy alone
    does not decide which model to deploy -- accuracy per operation does.
    We count only the layers that dominate (conv and linear); activations,
    pooling and normalisation are negligible by comparison.
    """
    macs = [0]
    hooks = []

    def conv_hook(m, inp, out):
        # each output element costs (in_ch/groups * k) multiply-accumulates
        k = int(np.prod(m.kernel_size))
        macs[0] += out.numel() * (m.in_channels // m.groups) * k

    def lin_hook(m, inp, out):
        macs[0] += m.in_features * m.out_features * out.shape[0]

    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))

    model.eval()
    with torch.no_grad():
        model(torch.zeros(*input_shape))
    for h in hooks:
        h.remove()
    return macs[0]


def set_all_seeds(seed=config.SEED):
    """
    Fix every random number generator we use, so re-running gives the same
    answer. Randomness enters in three places:
        1. numpy   -> the train/val/test split
        2. python  -> miscellaneous
        3. torch   -> weight initialisation and dropout
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(kind, **kwargs):
    """Factory so the training code does not need to know the class names."""
    return {"cnn": ConstellationCNN,
            "iqcnn": IQCNN,
            "mlp": FeatureMLP}[kind](**kwargs)


MODEL_INPUT_SHAPE = {
    "cnn":   (1, 1, config.IMG_SIZE, config.IMG_SIZE),
    "iqcnn": (1, 2, config.N_IQ_SYMBOLS),
    "mlp":   (1, 15),
}


def model_summary(model, input_shape):
    """Layer-by-layer output shapes and parameter counts, for the paper."""
    lines = ["{:<26s} {:>20s} {:>12s}".format("layer", "output shape", "params"),
             "-" * 60]
    x = torch.zeros(*input_shape)
    lines.append("{:<26s} {:>20s} {:>12s}".format(
        "input", str(tuple(x.shape[1:])), "0"))

    model.eval()
    with torch.no_grad():
        for top_name, top in model.named_children():
            mods = ([(top_name, top)] if not isinstance(top, nn.Sequential)
                    else [("{}.{}".format(top_name, n), m)
                          for n, m in top.named_children()])
            for name, m in mods:
                if isinstance(m, nn.Sequential):
                    for sn, sm in m.named_children():
                        x = sm(x)
                        lines.append("{:<26s} {:>20s} {:>12,d}".format(
                            "{}.{}".format(name, type(sm).__name__),
                            str(tuple(x.shape[1:])),
                            sum(q.numel() for q in sm.parameters())))
                else:
                    x = m(x)
                    lines.append("{:<26s} {:>20s} {:>12,d}".format(
                        "{}.{}".format(name, type(m).__name__),
                        str(tuple(x.shape[1:])),
                        sum(q.numel() for q in m.parameters())))
    lines.append("-" * 60)
    lines.append("{:<26s} {:>20s} {:>12,d}".format(
        "TOTAL", "", count_parameters(model)))
    return "\n".join(lines)


if __name__ == "__main__":
    set_all_seeds()
    print("=" * 70)
    print("  THE THREE NETWORKS ({} classes)".format(config.N_CLASSES))
    print("=" * 70)

    for kind, label in (("cnn", "2-D CNN on 64x64 constellation images"),
                        ("iqcnn", "1-D CNN on 512 raw I/Q samples"),
                        ("mlp", "MLP on 15 statistical features")):
        m = build_model(kind)
        shape = MODEL_INPUT_SHAPE[kind]
        print("\n" + "-" * 70)
        print("  {}  --  {}".format(kind.upper(), label))
        print("-" * 70)
        print(model_summary(m, shape))
        print("  MACs per inference: {:,}".format(count_flops(m, shape)))
        out = m(torch.randn(*((4,) + shape[1:])))
        print("  forward test: {} -> {}  OK".format(
            tuple((4,) + shape[1:]), tuple(out.shape)))
