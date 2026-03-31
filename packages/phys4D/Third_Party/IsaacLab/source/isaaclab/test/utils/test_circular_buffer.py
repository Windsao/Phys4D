




import torch

import pytest

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher


simulation_app = AppLauncher(headless=True).app

"""Rest everything follows from here."""

from isaaclab.utils import CircularBuffer


@pytest.fixture
def circular_buffer():
    """Create a circular buffer for testing."""
    max_len = 5
    batch_size = 3
    device = "cpu"
    return CircularBuffer(max_len, batch_size, device)


def test_initialization(circular_buffer):
    """Test initialization of the circular buffer."""
    assert circular_buffer.max_length == 5
    assert circular_buffer.batch_size == 3
    assert circular_buffer.device == "cpu"
    assert circular_buffer.current_length.tolist() == [0, 0, 0]


def test_reset(circular_buffer):
    """Test resetting the circular buffer."""

    data = torch.ones((circular_buffer.batch_size, 2), device=circular_buffer.device)
    circular_buffer.append(data)

    circular_buffer.reset()


    assert circular_buffer.current_length.tolist() == [0, 0, 0]


def test_reset_subset(circular_buffer):
    """Test resetting a subset of batches in the circular buffer."""
    data1 = torch.ones((circular_buffer.batch_size, 2), device=circular_buffer.device)
    data2 = 2.0 * data1.clone()
    data3 = 3.0 * data1.clone()
    circular_buffer.append(data1)
    circular_buffer.append(data2)

    reset_batch_id = 1
    circular_buffer.reset(batch_ids=[reset_batch_id])

    assert circular_buffer.current_length.tolist()[reset_batch_id] == 0

    circular_buffer.append(data3)

    expected_length = [3, 3, 3]
    expected_length[reset_batch_id] = 1
    assert circular_buffer.current_length.tolist() == expected_length

    for i in range(circular_buffer.max_length):
        torch.testing.assert_close(circular_buffer.buffer[reset_batch_id, 0], circular_buffer.buffer[reset_batch_id, i])


def test_append_and_retrieve(circular_buffer):
    """Test appending and retrieving data from the circular buffer."""

    data1 = torch.tensor([[1, 1], [1, 1], [1, 1]], device=circular_buffer.device)
    data2 = torch.tensor([[2, 2], [2, 2], [2, 2]], device=circular_buffer.device)

    circular_buffer.append(data1)
    circular_buffer.append(data2)

    assert circular_buffer.current_length.tolist() == [2, 2, 2]

    retrieved_data = circular_buffer[torch.tensor([0, 0, 0], device=circular_buffer.device)]
    assert torch.equal(retrieved_data, data2)

    retrieved_data = circular_buffer[torch.tensor([1, 1, 1], device=circular_buffer.device)]
    assert torch.equal(retrieved_data, data1)


def test_buffer_overflow(circular_buffer):
    """Test buffer overflow.

    If the buffer is full, the oldest data should be overwritten.
    """

    for count in range(circular_buffer.max_length + 2):
        data = torch.full((circular_buffer.batch_size, 4), count, device=circular_buffer.device)
        circular_buffer.append(data)


    assert circular_buffer.current_length.tolist() == [
        circular_buffer.max_length,
        circular_buffer.max_length,
        circular_buffer.max_length,
    ]


    key = torch.tensor([0, 0, 0], device=circular_buffer.device)
    retrieved_data = circular_buffer[key]
    expected_data = torch.full_like(data, circular_buffer.max_length + 1)

    assert torch.equal(retrieved_data, expected_data)


    key = torch.tensor(
        [circular_buffer.max_length - 1, circular_buffer.max_length - 1, circular_buffer.max_length - 1],
        device=circular_buffer.device,
    )
    retrieved_data = circular_buffer[key]
    expected_data = torch.full_like(data, 2)

    assert torch.equal(retrieved_data, expected_data)


def test_empty_buffer_access(circular_buffer):
    """Test accessing an empty buffer."""
    with pytest.raises(RuntimeError):
        circular_buffer[torch.tensor([0, 0, 0], device=circular_buffer.device)]


def test_invalid_batch_size(circular_buffer):
    """Test appending data with an invalid batch size."""
    data = torch.ones((circular_buffer.batch_size + 1, 2), device=circular_buffer.device)
    with pytest.raises(ValueError):
        circular_buffer.append(data)

    with pytest.raises(ValueError):
        circular_buffer[torch.tensor([0, 0], device=circular_buffer.device)]


def test_key_greater_than_pushes(circular_buffer):
    """Test retrieving data with a key greater than the number of pushes.

    In this case, the oldest data should be returned.
    """
    data1 = torch.tensor([[1, 1], [1, 1], [1, 1]], device=circular_buffer.device)
    data2 = torch.tensor([[2, 2], [2, 2], [2, 2]], device=circular_buffer.device)

    circular_buffer.append(data1)
    circular_buffer.append(data2)

    retrieved_data = circular_buffer[torch.tensor([5, 5, 5], device=circular_buffer.device)]
    assert torch.equal(retrieved_data, data1)


def test_return_buffer_prop(circular_buffer):
    """Test retrieving the whole buffer for correct size and contents.
    Returning the whole buffer should have the shape [batch_size,max_len,data.shape[1:]]
    """
    num_overflow = 2
    for i in range(circular_buffer.max_length + num_overflow):
        data = torch.tensor([[i]], device=circular_buffer.device).repeat(3, 2)
        circular_buffer.append(data)

    retrieved_buffer = circular_buffer.buffer

    assert retrieved_buffer.shape == torch.Size([circular_buffer.batch_size, circular_buffer.max_length, 2])

    torch.testing.assert_close(retrieved_buffer[0], retrieved_buffer[1])

    torch.testing.assert_close(
        retrieved_buffer[:, 0], torch.tensor([[num_overflow]], device=circular_buffer.device).repeat(3, 2)
    )

    torch.testing.assert_close(
        retrieved_buffer[:, -1],
        torch.tensor([[circular_buffer.max_length + num_overflow - 1]], device=circular_buffer.device).repeat(3, 2),
    )

    for idx in range(circular_buffer.max_length - 1):
        assert torch.all(torch.le(retrieved_buffer[:, idx], retrieved_buffer[:, idx + 1]))
