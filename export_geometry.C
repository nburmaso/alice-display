// Export selected ALICE detector volumes as OBJ meshes for Blender
// Run in O2: root -l -b -q 'export_geometry.C++(568422,"display")'
// Blender coordinates (meters) = (ALICE Z, ALICE X, ALICE Y) / 100

#ifdef __CLING__
#pragma cling load("libEve")
#endif
#include <CCDB/BasicCCDBManager.h>
#include <DataFormatsParameters/AggregatedRunInfo.h>
#include <TGeoManager.h>
#include <TGeoNode.h>
#include <TGeoVolume.h>
#include <TGeoMatrix.h>
#include <TGeoShape.h>
#include <TGeoCompositeShape.h>
#include <TGeoShapeAssembly.h>
#include <TGeoMaterial.h>
#include <TGeoMedium.h>
#include <TBuffer3D.h>
#include <TEveGeoPolyShape.h>
#include <algorithm>
#include <cctype>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

// Compare complete node-path prefixes
bool under(const std::string& path, const std::string& root) {
  return path == root || (path.size() > root.size() && path.compare(0, root.size(), root) == 0 && path[root.size()] == '/');
}

bool oneOf(const std::string& name, std::initializer_list<const char*> names) {
  for (auto* value : names) if (name == value) return true;
  return false;
}

// TBuffer3D stores polygons as unordered segment IDs, not vertex indices
// Follow the connected segments to recover an ordered polygon boundary
bool polygonLoop(const TBuffer3D& mesh, int count, const int* ids, std::vector<int>& ring) {
  if (count < 3 || count > 5000 || ids[0] < 0 || ids[0] >= int(mesh.NbSegs())) return false;
  ring.clear();
  std::vector<bool> used(count, false);
  int first = mesh.fSegs[3 * ids[0] + 1];
  int current = mesh.fSegs[3 * ids[0] + 2];
  ring.push_back(first);
  ring.push_back(current);
  used[0] = true;
  for (int n = 1; n < count; ++n) {
    bool found = false;
    for (int j = 1; j < count; ++j) {
      if (used[j] || ids[j] < 0 || ids[j] >= int(mesh.NbSegs())) continue;
      int a = mesh.fSegs[3 * ids[j] + 1], b = mesh.fSegs[3 * ids[j] + 2];
      if (a != current && b != current) continue;
      current = (a == current) ? b : a;
      ring.push_back(current);
      used[j] = true;
      found = true;
      break;
    }
    if (!found) return false;
  }
  if (ring.back() != first) return false;
  ring.pop_back();
  for (int v : ring) if (v < 0 || v >= int(mesh.NbPnts())) return false;
  return ring.size() >= 3;
}

// Tessellate a placed TGeo solid and append it to one OBJ file
// Boolean solids need TEveGeoPolyShape
bool writeObj(const TGeoShape* shape, const TGeoHMatrix& transform,
              std::ofstream& out, unsigned long long& vertexIndex, bool& composite, int segments) {
  if (dynamic_cast<const TGeoShapeAssembly*>(shape)) return false;
  std::unique_ptr<TEveGeoPolyShape> csgMesh;
  if (auto* csg = dynamic_cast<const TGeoCompositeShape*>(shape)) {
    composite = true;
    csgMesh.reset(TEveGeoPolyShape::Construct(const_cast<TGeoCompositeShape*>(csg), segments));
    if (!csgMesh) return false;
    shape = csgMesh.get();
  }
  std::unique_ptr<TBuffer3D> mesh(shape->MakeBuffer3D());
  if (!mesh || !mesh->NbPnts() || !mesh->NbPols()) return false;

  // Resolve polygons before writing vertices so a failed shape leaves no orphan data.
  std::vector<std::vector<int>> polygons;
  std::vector<int> ring;
  int pos = 0;
  for (unsigned p = 0; p < mesh->NbPols(); ++p) {
    int n = mesh->fPols[pos + 1];
    if (n < 3 || n > 5000) break;
    if (polygonLoop(*mesh, n, mesh->fPols + pos + 2, ring)) polygons.push_back(ring);
    pos += n + 2;
  }
  if (polygons.empty()) return false;

  for (unsigned i = 0; i < mesh->NbPnts(); ++i) {
    double local[3] = {mesh->fPnts[3*i], mesh->fPnts[3*i+1], mesh->fPnts[3*i+2]};
    double global[3];
    transform.LocalToMaster(local, global);
    out << "v " << global[2]*.01 << ' ' << global[0]*.01 << ' ' << global[1]*.01 << '\n';
  }
  for (const auto& face : polygons) {
    out << 'f';
    for (int i : face) out << ' ' << vertexIndex + i;
    out << '\n';
  }
  vertexIndex += mesh->NbPnts();
  return true;
}

// Map actual O2 volume names and their parent paths to Blender rendering layers
// A leading '@' means: carry the label through children and export the leaves
std::string chosenGroup(const std::string& path, const std::string& vol,
                        const std::string& inherited, int children, const std::string& medium) {
  const std::string barrel = "cave_1/barrel_1";
  const std::string trdtof = barrel + "/B077_1";

  // EMCal/DCal: steel modules or scintillator cells
  if (vol == "EMOD" || vol == "SCMX") {
    const std::string prefix = barrel + "/";
    if (path.compare(0, prefix.size(), prefix) == 0) {
      const auto slash = path.find('/', prefix.size());
      if (slash != std::string::npos) {
        const std::string placement = path.substr(prefix.size(), slash-prefix.size());
        const std::string mother = placement.substr(0, placement.find('_'));
        if (oneOf(mother, {"SMOD", "SM10", "SM3rd"})) return vol == "SCMX" ? "EMCAL_CELLS" : "EMCAL";
        if (oneOf(mother, {"DCSM", "DCEXT"})) return vol == "SCMX" ? "DCAL_CELLS" : "DCAL";
      }
    }
  }
  if (under(path, barrel + "/TPC_M_1")) {
    if (oneOf(vol, {"TPC_IFC", "TPC_OFCSEG"})) return "@TPC_CAGE";
    if (vol == "TPC_SSWHEEL") return "@TPC_ENDCAP";
    if (oneOf(vol, {"TPC_ICVM", "TPC_ICVS", "TPC_HSM", "TPC_HSS", "TPC_CDRS", "TPC_CDCI", "TPC_CDCE"})) return "TPC_SHELL";
    if (vol == "TPC_Drift") return "TPC_ACTIVE";
  }
  if (under(path, "cave_1/Dipole_1/DYoke_1")) return children ? "@DIPOLE_YOKE" : "DIPOLE_YOKE";
  if (under(path, "cave_1/Dipole_1/DCoil_1")) return children ? "@DIPOLE_COIL" : "DIPOLE_COIL";
  if (under(path, barrel + "/AFA_1") && oneOf(vol, {"AFaGraphiteCone", "AFaConcreteCone", "AFaSteelCone25", "AFaSteelCone31", "AFaPbCone", "AFaCH2Cone", "AFaWTube1", "AFaWTube2", "AFaWTube3", "AFaWTube4", "AFaWTube5"})) return "FRONT_ABSORBER";
  if (under(path, "cave_1/YOUT2_1") && vol == "YMuonFilter") return "MUON_FILTER";
  if (under(path, "cave_1/YSAA_1") && oneOf(vol, {"YSAA1_WCone", "YSAA1_WPipe", "YSAA1_PbComp", "YSAA2_PbComp", "YSAA2_PbRing", "YSAA2_SteelEnvelope", "YSAA3CCBlock"})) return "BEAM_SHIELD";
  if (under(path, barrel + "/L3MO_1") && oneOf(vol, {"L3TS", "L3YO", "L3DoorR", "L3DoorL"})) return "L3_MAGNET";
  if (under(path, barrel) && oneOf(vol, {"IP_PIPE", "aluBeforeBellows", "votube4", "voadaptator", "aluSideA", "steelFlangeSideA", "RB26s12Tube", "voRB24CuTube", "voRB24CuTubeA", "RB24IpShell"})) return "BEAM_PIPE";

  // Sensitive detector layers, with path-scoped names to avoid false matches.
  if (under(path, barrel + "/ITSV_2") && vol.size() == 11 && vol.rfind("ITSUSensor",0) == 0 && vol.back() >= '0' && vol.back() <= '6') return "ITS";
  if (under(path, barrel + "/MFT_0") && vol == "MFTSensor") return "MFT";
  if (under(path, trdtof) && vol.size() == 4 && vol[0] == 'U' && (vol[1] == 'J' || vol[1] == 'K') && std::isdigit((unsigned char)vol[2]) && std::isdigit((unsigned char)vol[3])) return "TRD";
  if (under(path, trdtof) && vol == "FSEN") return "TOF";
  if (under(path, barrel + "/FT0A_1") && vol == "0REG") return "FT0A";
  if (under(path, barrel + "/FT0C_1") && vol == "0REG") return "FT0C";
  if (under(path, barrel + "/FV0_1") && medium == "FV0_Scintillator$") return "FV0A";
  if (under(path, "cave_1/caveRB24_1") && vol == "FDApad") return "FDDA";
  if ((under(path, "cave_1/FDC_1") || under(path, "cave_1/FDC_2")) && vol == "FDCpad") return "FDDC";

  // MCH stations occupy YOUT1, DDIP and YOUT2; MID RPCs are in YOUT2.
  if (under(path, "cave_1/YOUT1_1") || under(path, "cave_1/DDIP_1") || under(path, "cave_1/YOUT2_1")) {
    const bool mchGas = oneOf(vol, {"SA1G", "SA2G", "Segment 0 gas", "Segment 1 gas", "Segment 2 gas"}) ||
                        (vol.size() > 4 && vol.compare(vol.size()-4, 4, " gas") == 0);
    if (mchGas) return "MCH";
  }
  if (under(path, "cave_1/YOUT2_1") && vol.rfind("Gas ",0) == 0 && vol.find("RPC") != std::string::npos) return "MID";

  if (!inherited.empty() && inherited[0] == '@' && !children) return inherited.substr(1);
  return inherited;
}

// run: ALICE run number, used to retrieve the corresponding aligned TGeo geometry
// prefix: files are named <prefix>_<GROUP>.obj; groups: comma-separated layer names
// maxMeshes: limit exported placements; nSegments: curved-solid tessellation resolution
void export_geometry(int run, const char* prefix = "display",
                     const char* groups = "EMCAL,DCAL,L3_MAGNET,TPC_CAGE,TPC_ENDCAP,TPC_SHELL,DIPOLE_COIL,FRONT_ABSORBER,MUON_FILTER,BEAM_SHIELD,BEAM_PIPE,MFT,FT0A,FT0C,FV0A,MCH,MID,ITS,TPC_ACTIVE,TRD,TOF",
                     int maxMeshes = 100000, int nSegments = 64,
                     const char* ccdbURL = "http://alice-ccdb.cern.ch") {
  auto& ccdb = o2::ccdb::BasicCCDBManager::instance();
  ccdb.setURL(ccdbURL);
  ccdb.setCaching(true);
  ccdb.setLocalObjectValidityChecking();
  const auto info = o2::parameters::AggregatedRunInfo::buildAggregatedRunInfo(ccdb.instance(), run);
  const auto timestamp = info.sor/2 + info.eor/2;
  auto* geo = ccdb.getForTimeStamp<TGeoManager>("GLO/Config/GeometryAligned", timestamp);
  geo->SetNsegments(nSegments);
  std::cout << "Aligned geometry for run " << run << ", timestamp " << timestamp << " ms\n";

  // 2. Choose the geometry layers and the relevant TGeo subtrees.
  std::set<std::string> enabled;
  std::string group;
  for (char c : std::string(groups) + ",") {
    if (c == ',') { if (!group.empty()) enabled.insert(group); group.clear(); }
    else if (c != ' ') group += c;
  }
  auto need = [&](std::initializer_list<const char*> names) {
    for (auto* name : names) if (enabled.count(name)) return true;
    return false;
  };
  std::vector<std::string> roots;
  if (need({"TPC_CAGE","TPC_ENDCAP","TPC_SHELL","TPC_ACTIVE"})) roots.emplace_back("cave_1/barrel_1/TPC_M_1");
  if (need({"FRONT_ABSORBER"})) roots.emplace_back("cave_1/barrel_1/AFA_1");
  if (need({"DIPOLE_YOKE","DIPOLE_COIL"})) roots.emplace_back("cave_1/Dipole_1");
  if (need({"MUON_FILTER","MCH","MID"})) roots.emplace_back("cave_1/YOUT2_1");
  if (need({"BEAM_SHIELD"})) roots.emplace_back("cave_1/YSAA_1");
  if (need({"L3_MAGNET"})) roots.emplace_back("cave_1/barrel_1/L3MO_1");
  if (need({"BEAM_PIPE"})) for (auto* path : {"BeamPipeCsideSection_1","aluSideA_1","steelFlangeSideA_1","RB24_1","RB26s12TubeM_1"})
    roots.emplace_back(std::string("cave_1/barrel_1/")+path);
  if (need({"ITS"})) roots.emplace_back("cave_1/barrel_1/ITSV_2");
  if (need({"MFT"})) roots.emplace_back("cave_1/barrel_1/MFT_0");
  if (need({"TRD","TOF"})) roots.emplace_back("cave_1/barrel_1/B077_1");
  if (need({"FT0A"})) roots.emplace_back("cave_1/barrel_1/FT0A_1");
  if (need({"FT0C"})) roots.emplace_back("cave_1/barrel_1/FT0C_1");
  if (need({"FV0A"})) roots.emplace_back("cave_1/barrel_1/FV0_1");
  if (need({"FDDA"})) roots.emplace_back("cave_1/caveRB24_1");
  if (need({"FDDC"})) { roots.emplace_back("cave_1/FDC_1"); roots.emplace_back("cave_1/FDC_2"); }
  if (need({"MCH"})) { roots.emplace_back("cave_1/YOUT1_1"); roots.emplace_back("cave_1/DDIP_1"); }

  // EMCal and DCal supermodule placements have instance-specific names
  if (need({"EMCAL","EMCAL_CELLS","DCAL","DCAL_CELLS"})) {
    auto* cave = geo->GetTopNode()->GetVolume();
    for (int i = 0; i < cave->GetNdaughters(); ++i) {
      auto* barrel = cave->GetNode(i);
      if (std::string(barrel->GetName()) != "barrel_1") continue;
      auto* volume = barrel->GetVolume();
      for (int j = 0; j < volume->GetNdaughters(); ++j) {
        const std::string placement = volume->GetNode(j)->GetName();
        const std::string mother = placement.substr(0, placement.find('_'));
        if ((oneOf(mother, {"SMOD","SM10","SM3rd"}) && need({"EMCAL","EMCAL_CELLS"})) ||
            (oneOf(mother, {"DCSM","DCEXT"}) && need({"DCAL","DCAL_CELLS"})))
          roots.emplace_back("cave_1/barrel_1/"+placement);
      }
      break;
    }
  }

  // Visit only selected branches, accumulating the placement transform
  std::map<std::string, std::ofstream> files;
  std::map<std::string, unsigned long long> nextVertex;
  std::map<std::string, int> counts;
  int written = 0, composites = 0, failures = 0;
  std::function<void(TGeoNode*, const TGeoHMatrix&, const std::string&, const std::string&)> visit;
  visit = [&](TGeoNode* node, const TGeoHMatrix& parent, const std::string& parentPath, const std::string& inherited) {
    if (written >= maxMeshes) return;
    const std::string path = parentPath.empty() ? node->GetName() : parentPath + "/" + node->GetName();
    bool relevant = false;
    for (const auto& root : roots) if (under(path,root) || under(root,path)) { relevant = true; break; }
    if (!relevant) return;  // critical: do not traverse millions of unrelated volumes

    auto* volume = node->GetVolume();
    TGeoHMatrix global(parent);
    if (node->GetMatrix()) global.Multiply(node->GetMatrix());
    auto* medium = volume->GetMedium();
    std::string label = chosenGroup(path, volume->GetName(), inherited, volume->GetNdaughters(), medium ? medium->GetName() : "");
    bool subtree = !label.empty() && label[0] == '@';
    std::string selected = subtree ? label.substr(1) : label;

    if (!subtree && enabled.count(selected)) {
      auto* material = volume->GetMaterial();
      const std::string matName = material ? material->GetName() : "";
      const std::string medName = medium ? medium->GetName() : "";
      if (matName.find("AIR") == std::string::npos && matName.find("VACUUM") == std::string::npos &&
          medName.find("Air$") == std::string::npos && medName.find("Vacuum$") == std::string::npos) {
        if (!files.count(selected)) {
          files[selected].open(std::string(prefix)+"_"+selected+".obj");
          files[selected] << "# ALICE geometry (m), Blender XYZ=(ALICE Z,X,Y)/100\n";
          nextVertex[selected] = 1;
        }
        bool composite = false;
        if (writeObj(volume->GetShape(), global, files[selected], nextVertex[selected], composite, nSegments)) {
          ++written;
          ++counts[selected];
          if (composite) ++composites;
        } else ++failures;
      }
      // Keep the selected top-level solid, descend only for separately requested cells
      if (selected != "TPC_ACTIVE" &&
          !(selected == "EMCAL" && enabled.count("EMCAL_CELLS")) &&
          !(selected == "DCAL" && enabled.count("DCAL_CELLS"))) return;
    }
    const std::string next = subtree ? label : (!inherited.empty() && inherited[0] == '@' ? inherited : "");
    for (int i = 0; i < volume->GetNdaughters(); ++i)
      visit(volume->GetNode(i), global, path, next);
  };
  visit(geo->GetTopNode(), TGeoHMatrix(), "", "");

  // 4. Summarize exported rendering layers.
  std::cout << "Exported " << written << " placements (" << composites << " Boolean solids).\n";
  for (const auto& name : enabled) std::cout << "  " << name << ": " << counts[name] << '\n';
  std::cout << "Tessellation failures: " << failures << '\n';
  if (written >= maxMeshes) std::cout << "maxMeshes reached; output truncated\n";
}
