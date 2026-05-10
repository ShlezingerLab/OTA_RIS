---
name: Channel Sampling Modification Plan
overview: Modify the channel matching loss computation to randomly sample a subset of channels per batch sample, instead of using all channels. This will reduce memory usage and provide stochastic channel training.
todos: []
isProject: false
---

# Channel Sampling Modification Plan

## Changes Overview

Modify the teacher training to randomly sample a subset of channels for each batch sample during channel matching loss computation, rather than using all available channels.

## Current Behavior

In `[teachers.py](teachers.py)`, the `get_channel_matching_loss()` method (lines 712-771) currently:

- Takes all channels: `H_d` (N_ch, Nr, Nt), `H_1` (N_ch, Nt, Nm), `H_2` (N_ch, Nm, Nr)
- Expands each batch sample across ALL channels
- Total comparisons: `Batch_Size * Num_Channels`

```python
# Lines 736-743: Current expansion logic
s_expanded = s.repeat_interleave(num_channels, dim=0)  # Each sample uses ALL channels
y_expanded = y_learned.repeat_interleave(num_channels, dim=0)
H_d_expanded = H_d.repeat(batch_size, 1, 1)
H_1_expanded = H_1.repeat(batch_size, 1, 1)
H_2_expanded = H_2.repeat(batch_size, 1, 1)
```

## Proposed Changes

### 1. Modify `get_channel_matching_loss()` in `MyTeacher` class

Add new parameter `num_channels_sample` and implement random sampling:

**Location**: Line 712-771 in `[teachers.py](teachers.py)`

**New signature**:

```python
def get_channel_matching_loss(
    self,
    H_d: torch.Tensor,   # (N_ch_pool, Nr, Nt)
    H_1: torch.Tensor,   # (N_ch_pool, Nt, Nm)
    H_2: torch.Tensor,   # (N_ch_pool, Nm, Nr)
    num_channels_sample: int = None  # How many channels to sample per batch sample
) -> torch.Tensor:
```

**Implementation logic**:

- If `num_channels_sample` is None or >= total channels, use current behavior (all channels)
- Otherwise, for each batch sample, randomly sample `num_channels_sample` indices
- Create sampled channel tensors using the random indices
- Proceed with existing expansion and loss computation

**Pseudocode**:

```python
batch_size = s.size(0)
num_channels_pool = H_d.size(0)

if num_channels_sample is None or num_channels_sample >= num_channels_pool:
    # Use all channels (current behavior)
    num_channels_sample = num_channels_pool

# Sample channels: each batch sample gets different random channels
channel_indices = torch.randint(
    0, num_channels_pool,
    (batch_size, num_channels_sample),
    device=H_d.device
)

# Gather sampled channels for each batch sample
# Shape: (batch_size, num_channels_sample, ...)
H_d_sampled = H_d[channel_indices.view(-1)].view(batch_size, num_channels_sample, ...)
H_1_sampled = H_1[channel_indices.view(-1)].view(batch_size, num_channels_sample, ...)
H_2_sampled = H_2[channel_indices.view(-1)].view(batch_size, num_channels_sample, ...)

# Flatten to (batch_size * num_channels_sample, ...)
# Then proceed with existing logic
```

### 2. Update `train_teacher()` function signature

**Location**: Line 932 in `[teachers.py](teachers.py)`

Add new parameter:

```python
def train_teacher(teacher, train_loader, device, epochs, lr, weight_decay, lambda_l2=0.0,
                 H_d_channel=None, H_1_channel=None, H_2_channel=None,
                 lambda_class=0.0,
                 num_channels_sample=None,  # NEW PARAMETER
                 save_path=None, wandb_run=None):
```

**Pass to loss function** (around line 987):

```python
loss_channel = teacher.get_channel_matching_loss(
    H_d_channel, H_1_channel, H_2_channel,
    num_channels_sample=num_channels_sample
)
```

### 3. Update main script (**main** section)

**Location**: Lines 1169-1175 in `[teachers.py](teachers.py)`

Add new configuration parameter and pass to training:

```python
# Add to configuration section (around line 1095)
num_channels_sample = 100  # Sample 100 channels per batch sample

# Pass to train_teacher call (line 1169)
train_teacher(teacher, train_loader, device, epochs, lr, weight_decay,
            H_d_channel=H_d_all,
            H_1_channel=H_1_all,
            H_2_channel=H_2_all,
            lambda_class=lambda_class,
            num_channels_sample=num_channels_sample,  # NEW
            save_path=save_path,
            wandb_run=run)
```

### 4. Optional: Add command-line argument

**Location**: Lines 1074-1078 in `[teachers.py](teachers.py)`

```python
parser.add_argument('--num_channels_sample', type=int, default=None,
                   help='Number of channels to sample per batch sample (None = use all)')
```

Then use: `num_channels_sample = args.num_channels_sample`

## Benefits

- **Memory Efficiency**: Reduces GPU memory from `B * 6000` to `B * num_channels_sample` comparisons
- **Faster Training**: Fewer channel evaluations per batch
- **Stochastic Training**: Each sample sees different channel realizations, potentially better generalization
- **Flexibility**: Can tune the trade-off between computational cost and channel diversity

## Example

With batch_size=256, channel_pool=6000, num_channels_sample=100:

- **Before**: 256 * 6000 = 1,536,000 comparisons
- **After**: 256 * 100 = 25,600 comparisons (~60x reduction)
