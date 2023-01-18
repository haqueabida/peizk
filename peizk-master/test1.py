import secrets
import math
from helpfunctions import hash_to_prime, is_prime, shamir_trick
from main import setup, add, prove_membership, delete, verify_membership, \
        prove_membership_with_NIPoE, verify_exponentiation, batch_prove_membership, batch_verify_membership, \
        batch_prove_membership_with_NIPoE, batch_verify_membership_with_NIPoE, batch_add, \
        prove_non_membership, verify_non_membership, batch_delete_using_membership_proofs,\
        create_all_membership_witnesses, aggregate_membership_witnesses, calculate_product


