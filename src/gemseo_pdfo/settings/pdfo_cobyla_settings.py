# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""Settings for the PDFO COBYLA algorithm."""

from __future__ import annotations

from gemseo_pdfo.settings.base_pdfo_settings import BasePDFOSettings


class PDFO_COBYLA_Settings(BasePDFOSettings):  # noqa: N801
    """The PDFO COBYLA setting."""

    # TODO(bump-gemseo): read it with the target_class_name property of a settings instance; in a settings class, the value is now derived from the name of the class (X_Settings targets X), so remove the assignment if its value is the name of the class without _Settings, else rename the class to <value>_Settings first, or it would target another class  # noqa: E501
    _TARGET_CLASS_NAME = "PDFO_COBYLA"
