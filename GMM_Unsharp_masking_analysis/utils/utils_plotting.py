from pyregion.mpl_helper import properties_func_default


def set_color(color):
    def fixed_color(shape, saved_attrs):
        attr_list, attr_dict = saved_attrs
        attr_dict["color"] = color
        kwargs = properties_func_default(shape, (attr_list, attr_dict))

        return kwargs

    return fixed_color


def set_prop(color, lw=1):
    def fixed_prop(shape, saved_attrs):
        attr_list, attr_dict = saved_attrs
        attr_dict["color"] = color
        attr_dict["width"] = lw
        kwargs = properties_func_default(shape, (attr_list, attr_dict))

        return kwargs

    return fixed_prop


def ax_pos(ncol: int, nrow: int, i: int) -> list[float]:
    return [
        i % ncol * 1 / ncol,
        1 - i // ncol * 1 / nrow - 1 / nrow,
        1 / ncol - 0.01,
        1 / nrow - 0.01,
    ]
