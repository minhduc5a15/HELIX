#include "autograd/autograd_meta.hpp"

#include "core/autograd_meta.hpp"

namespace helix {

    void AutogradMetaDeleter::operator()(AutogradMeta* meta) const { delete meta; }

}  // namespace helix
