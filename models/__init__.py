from .group_matcher import build_group_matcher
from .criterion import SetCriterion
from .models import GADTR


def build_model(args):
    model = GADTR(args)

    losses = ['labels', 'cardinality']
    group_losses = ['group_labels', 'group_cardinality', 'group_code', 'group_consistency']

    # Set loss coefficients
    weight_dict = {}
    weight_dict['loss_ce'] = args.ce_loss_coef
    weight_dict['loss_group_ce'] = args.group_ce_loss_coef
    weight_dict['loss_group_code'] = args.group_code_loss_coef
    weight_dict['loss_consistency'] = args.consistency_loss_coef

    # Group matching
    group_matcher = build_group_matcher(args)

    # Loss functions
    criterion = SetCriterion(args.num_class, weight_dict=weight_dict, eos_coef=args.eos_coef,
                             losses=losses, group_losses=group_losses, group_matcher=group_matcher, args=args)

    criterion.cuda()

    return model, criterion
