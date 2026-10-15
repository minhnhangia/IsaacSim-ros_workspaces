# SPDX-License-Identifier: Apache-2.0
"""USD physics materials (friction) and their bindings."""

from pxr import PhysxSchema, UsdPhysics, UsdShade


def define_physics_material(stage, path: str, static_friction: float, dynamic_friction: float,
                            combine_mode: str = "multiply") -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr().Set(static_friction)
    api.CreateDynamicFrictionAttr().Set(dynamic_friction)
    api.CreateRestitutionAttr().Set(0.0)
    physx = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    physx.CreateFrictionCombineModeAttr().Set(combine_mode)
    physx.CreateRestitutionCombineModeAttr().Set(combine_mode)
    return material


def bind_physics_material(prim, material: UsdShade.Material) -> None:
    """Applies to the prim's colliders and its descendants', unless one has a physics material of its own."""
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")
