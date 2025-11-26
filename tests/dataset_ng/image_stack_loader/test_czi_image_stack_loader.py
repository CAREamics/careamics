import pytest

from careamics.config.validators import check_czi_axes_validity
from careamics.dataset_ng.image_stack.czi_image_stack import (
    CziImageStack,
)
from careamics.dataset_ng.image_stack_loader import load_czis


class TestCziImageStackLoader:

    @pytest.mark.czi
    @pytest.mark.parametrize(
        "axes",
        [
            # valid axes
            "ZYX",
            "CZYX",
            "TYX",
            "SCTYX",
            "SCZYX",
            "SCYX",
            # invalid axes
            "XZY",
            "TZYX",
            "SCXZY",
            "YX",
            "SCZT",
            "TCSYX",
        ],
    )
    def test_load_czis(self, mocker, axes):
        # mock CziImageStack.__init__ and CziImageStack.get_bounding_rectangles
        mock_init = mocker.patch.object(CziImageStack, "__init__", return_value=None)
        mock_get_bounding_rectangles = mocker.patch.object(
            CziImageStack,
            "get_bounding_rectangles",
            return_value={"mocked_bounding_rectangle": 0},
        )

        source = ["mocked_file.czi"]

        if not check_czi_axes_validity(axes):
            with pytest.raises(ValueError):
                _ = load_czis(source=source, axes=axes)
        else:
            _ = load_czis(source=source, axes=axes)
            mock_get_bounding_rectangles.assert_called()
            mock_init.assert_called()
