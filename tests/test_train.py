from click.testing import CliRunner
import pytest

from src.forest_cover_ml.logistic import train


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


def test_error_for_invalid_test_split_ratio(
        runner: CliRunner
) -> None:
    """It fails when test split ratio is greater than 1."""
    result = runner.invoke(
        train,
        [
            "--test-split-ratio",
            42,
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--test-split-ratio'" in result.output


def test_error_for_max_iter(
        runner: CliRunner
) -> None:
    result = runner.invoke(train, ["--tol", 0.0001])
    assert result.exit_code == 1


def test_error_for_random_state(
        runner: CliRunner
) -> None:
    result = runner.invoke(train, ["--random-state", 42])
    assert result.exit_code == 1


def test_error_for_C(
        runner: CliRunner
) -> None:
    result = runner.invoke(train, ["--logreg-c", 2.0])
    assert result.exit_code == 1
