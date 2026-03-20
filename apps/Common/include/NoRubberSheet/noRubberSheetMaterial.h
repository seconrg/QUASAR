#ifndef NO_RUBBER_SHEET_MATERIAL_H
#define NO_RUBBER_SHEET_MATERIAL_H

#include <Materials/Material.h>
#include <Materials/UnlitMaterial.h>

namespace quasar {

class NoRubberSheetMaterial : public UnlitMaterial {

public:
    NoRubberSheetMaterial(const UnlitMaterialCreateParams& params);
    ~NoRubberSheetMaterial() = default;

};

} // namespace quasar
#endif // NO_RUBBER_SHEET_MATERIAL_H