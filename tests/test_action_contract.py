def test_action_dims():
    from src.rlbot_integration.controller_adapter import CONT_DIM, DISC_DIM
    assert CONT_DIM == 5 and DISC_DIM == 3
