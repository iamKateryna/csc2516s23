import torch

def weighted_cross_entropy_with_logits(targets, logits, pos_weight):
    return targets * -logits.sigmoid().log() * pos_weight + (1 - targets) * -(1 - logits.sigmoid()).log()

def weighted_cross_entropy_with_logits_with_masked_class(
    pos_weight = 1.0):
  """Wrapper function for masked weighted cross-entropy with logits.
  This loss function ignores the classes with negative class id.
  Args:
    pos_weight: A coefficient to use on the positive examples.
  Returns:
    A weighted cross-entropy with logits loss function that ignores classes
    with negative class id.
  """

  def masked_weighted_cross_entropy_with_logits(y_true, logits):
    mask = (~y_true.eq(-1)).double()
    loss = torch.mean(mask * weighted_cross_entropy_with_logits(targets=y_true, logits=logits, pos_weight=pos_weight))
    return loss

  return masked_weighted_cross_entropy_with_logits