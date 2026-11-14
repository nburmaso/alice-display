// Select collision- or BC-based display candidates and export track trajectories
// ROOT coordinates are in ALICE centimeters
// root -l -b -q 'export_event.C++("AO2D.root","events","selection.conf")'

#include <TFile.h>
#include <TTree.h>
#include <TDirectory.h>
#include <TKey.h>
#include <TEnv.h>
#include <TSystem.h>
#include <TGeoManager.h>

#include <CCDB/BasicCCDBManager.h>
#include <DataFormatsParameters/AggregatedRunInfo.h>
#include <DataFormatsParameters/GRPMagField.h>
#include <DetectorsBase/GeometryManager.h>
#include <DetectorsBase/MatLayerCylSet.h>
#include <DetectorsBase/Propagator.h>
#include <MCHTracking/TrackExtrap.h>
#include <MCHTracking/TrackParam.h>
#include <ReconstructionDataFormats/TrackParametrization.h>
#include <CommonConstants/LHCConstants.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

constexpr unsigned kITS = 1, kTPC = 2, kTRD = 4, kTOF = 8;
constexpr unsigned kMFT = 16, kMCH = 32, kMID = 64;

// AO2D tables may have version suffixes such as O2bc_001
TTree* aodTree(TDirectory* df, const char* name)
{
  if (auto* tree = dynamic_cast<TTree*>(df->Get(name))) return tree;
  TIter keys(df->GetListOfKeys());
  while (auto* key = static_cast<TKey*>(keys())) {
    if (std::string(key->GetName()).rfind(std::string(name) + "_", 0) == 0)
      return dynamic_cast<TTree*>(df->Get(key->GetName()));
  }
  return nullptr;
}

void use(TTree* tree, const char* branch, void* address)
{
  tree->SetBranchStatus(branch, 1);
  tree->SetBranchAddress(branch, address);
}

unsigned detectorMaskFromNames(const std::string& names)
{
  static const std::map<std::string, unsigned> bits = {
    {"ITS", kITS}, {"TPC", kTPC}, {"TRD", kTRD}, {"TOF", kTOF},
    {"MFT", kMFT}, {"MCH", kMCH}, {"MID", kMID}
  };
  std::stringstream stream(names);
  std::string name;
  unsigned mask = 0;
  while (std::getline(stream, name, ',')) {
    name.erase(std::remove_if(name.begin(), name.end(), [](unsigned char c) {
      return std::isspace(c) || c == '"' || c == '\'';
    }), name.end());
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
      return std::toupper(c);
    });
    if (name.empty() || name == "NONE") continue;
    auto it = bits.find(name);
    if (it == bits.end()) throw std::invalid_argument("Unknown detector: " + name);
    mask |= it->second;
  }
  return mask;
}

// Sample the barrel track with the full magnetic field and material LUT
void sampleBarrel(float x, float y, float z, float alpha, float snp, float tgl,
                  float qpt, float maxLength, float maxChord, std::vector<float>& xyz)
{
  using Track = o2::track::TrackParametrization<float>;
  using MatCorr = o2::base::Propagator::MatCorrType;
  auto* propagator = o2::base::Propagator::Instance();
  Track reference(x, alpha, {y, z, snp, tgl, qpt});
  auto p0 = reference.getXYZGlo();
  std::array<float, 3> start = {p0.X(), p0.Y(), p0.Z()};
  const float radius = std::hypot(start[0], start[1]);
  std::vector<std::array<float, 3>> inward, outward;
  float length = 0;

  for (int direction : {-1, 1}) {
    auto track = reference;
    auto previous = start;
    auto& points = direction < 0 ? inward : outward;
    for (float r = radius + direction; r >= 2 && r <= 420; r += direction) {
      if (!propagator->propagateToR(track, r, false, 0.95f, 2.f, MatCorr::USEMatCorrLUT)) break;
      auto p = track.getXYZGlo();
      std::array<float, 3> next = {p.X(), p.Y(), p.Z()};
      const float step = std::hypot(next[0] - previous[0], next[1] - previous[1], next[2] - previous[2]);
      if (!(step <= maxChord && length + step <= maxLength) || std::abs(next[2]) > 500) break;
      points.push_back(next);
      length += step;
      previous = next;
    }
  }

  for (auto it = inward.rbegin(); it != inward.rend(); ++it)
    xyz.insert(xyz.end(), it->begin(), it->end());
  xyz.insert(xyz.end(), start.begin(), start.end());
  for (const auto& point : outward)
    xyz.insert(xyz.end(), point.begin(), point.end());
}

// Extrapolate the MCH track to the vertex and sample the spectrometer trajectory
void sampleForward(float x, float y, float z, float phi, float tgl, float qpt,
                   int type, const std::array<float, 3>& vertexXYZ,
                   const std::array<float, 2>& vertexCov, std::vector<float>& xyz)
{
  const double parameters[5] = {
    x, std::cos(phi) / tgl, y, std::sin(phi) / tgl, qpt / std::hypot(tgl, std::sin(phi))
  };
  o2::mch::TrackParam reference(z, parameters);
  auto append = [&](o2::mch::TrackParam& track) {
    xyz.insert(xyz.end(), {float(track.getNonBendingCoor()), float(track.getBendingCoor()), float(track.getZ())});
  };

  auto atVertex = reference;
  bool propagated = o2::mch::TrackExtrap::extrapToVertex(
    atVertex, vertexXYZ[0], vertexXYZ[1], vertexXYZ[2], vertexCov[0], vertexCov[1]);
  if (propagated) {
    auto track = atVertex;
    append(track);
    for (float zSample = vertexXYZ[2] - 1; zSample >= -90; zSample -= 1) {
      if (!o2::mch::TrackExtrap::extrapToZ(track, zSample)) break;
      append(track);
    }
  }

  // Approximate the path through the front absorber by joining its propagated endpoints
  auto absorberExit = reference;
  if (o2::mch::TrackExtrap::extrapToZ(absorberExit, -505.)) {
    if (propagated && xyz.size() >= 3) {
      const std::array<float, 3> a = {xyz[xyz.size()-3], xyz[xyz.size()-2], xyz.back()};
      const std::array<float, 3> b = {
        float(absorberExit.getNonBendingCoor()), float(absorberExit.getBendingCoor()), -505.f
      };
      for (int i = 1; i <= 16; ++i) {
        float f = i / 16.f;
        xyz.insert(xyz.end(), {a[0] + f * (b[0]-a[0]), a[1] + f * (b[1]-a[1]), a[2] + f * (b[2]-a[2])});
      }
    } else {
      append(absorberExit);
    }
    auto track = absorberExit;
    for (float zSample = -510; zSample > z; zSample -= 5) {
      if (!o2::mch::TrackExtrap::extrapToZ(track, zSample)) break;
      append(track);
    }
  }

  append(reference);
  auto track = reference;
  for (float zSample = z - 5; zSample >= -1465; zSample -= 5) {
    if (!o2::mch::TrackExtrap::extrapToZ(track, zSample)) break;
    append(track);
  }
  if (type == 3) {
    auto mid = reference;
    if (o2::mch::TrackExtrap::extrapToMID(mid)) {
      append(mid);
      track = mid;
    }
    for (float zSample = float(track.getZ()) - 2; zSample >= -1780; zSample -= 2) {
      if (!o2::mch::TrackExtrap::extrapToZ(track, zSample)) break;
      append(track);
    }
  }
}

// Select candidates before sampling trajectories
// mode: "collision" groups by collision ID, "bc" groups by assigned global BC
void export_event(const char* input = "AO2D.root", const char* outputDir = "events",
                  const char* config = "selection.conf", const char* ccdbURL = "http://alice-ccdb.cern.ch")
{
  TEnv conf(config);
  const std::string mode = conf.GetValue("Selection.Mode", "bc");
  const int nCandidates = conf.GetValue("Selection.Candidates", 5);
  const std::string onlyDF = conf.GetValue("Selection.Dataframe", "");
  const long long bcMin = std::stoll(conf.GetValue("Selection.BCMin", "-1"));
  const long long bcMax = std::stoll(conf.GetValue("Selection.BCMax", "-1"));
  const int minContrib = conf.GetValue("Collision.Contributors.Min", 0);
  const int maxContrib = conf.GetValue("Collision.Contributors.Max", 100000);
  const int minCount[3] = {conf.GetValue("Tracks.Total.Min", 2),
                           conf.GetValue("Tracks.Barrel.Min", 0), conf.GetValue("Tracks.Forward.Min", 0)};
  const int maxCount[3] = {conf.GetValue("Tracks.Total.Max", 10000),
                           conf.GetValue("Tracks.Barrel.Max", 10000), conf.GetValue("Tracks.Forward.Max", 10000)};
  const float ptMin[2] = {float(conf.GetValue("Barrel.Pt.Min", 0.2)), float(conf.GetValue("Forward.Pt.Min", 0.5))};
  const float ptMax[2] = {float(conf.GetValue("Barrel.Pt.Max", 10000.)), float(conf.GetValue("Forward.Pt.Max", 10000.))};
  const float etaMin[2] = {float(conf.GetValue("Barrel.Eta.Min", -0.9)), float(conf.GetValue("Forward.Eta.Min", -4.))};
  const float etaMax[2] = {float(conf.GetValue("Barrel.Eta.Max", 0.9)), float(conf.GetValue("Forward.Eta.Max", -2.5))};
  const unsigned required[2] = {detectorMaskFromNames(conf.GetValue("Barrel.RequireDet", "")),
                                detectorMaskFromNames(conf.GetValue("Forward.RequireDet", "MCH"))};
  const unsigned forbidden[2] = {detectorMaskFromNames(conf.GetValue("Barrel.VetoDet", "")),
                                 detectorMaskFromNames(conf.GetValue("Forward.VetoDet", ""))};
  const bool pvOnly = conf.GetValue("Barrel.RequirePVContributor", 0);
  const float maxLength = conf.GetValue("Barrel.MaxLengthCm", 600.);
  const float maxChord = conf.GetValue("Barrel.MaxChordCm", 10.);

  auto* aod = TFile::Open(input);
  auto& ccdb = o2::ccdb::BasicCCDBManager::instance();
  ccdb.setURL(ccdbURL);
  ccdb.setCaching(true);
  ccdb.setLocalObjectValidityChecking();
  gSystem->mkdir(outputDir, true);

  std::vector<std::string> dataframes;
  TIter keys(aod->GetListOfKeys());
  while (auto* key = static_cast<TKey*>(keys())) {
    if (std::string(key->GetName()).rfind("DF_", 0) == 0)
      dataframes.emplace_back(key->GetName());
  }

  int selected = 0, runCached = -1, dfIndex = -1;
  int64_t orbitResetUS = 0;
  for (const auto& dfName : dataframes) {
    if (selected == nCandidates) break;
    ++dfIndex;
    if (!onlyDF.empty() && onlyDF != dfName) continue;

    auto* df = aod->GetDirectory(dfName.c_str());
    auto* bcTree = aodTree(df, "O2bc");
    auto* collisions = aodTree(df, "O2collision");
    auto* barrel = aodTree(df, "O2track_iu");
    auto* extra = aodTree(df, "O2trackextra");
    auto* forward = aodTree(df, "O2fwdtrack");
    auto* ambiguousBarrel = aodTree(df, "O2ambiguoustrack");
    auto* ambiguousForward = aodTree(df, "O2ambiguousfwdtr");
    if (mode == "collision" && (!collisions || !collisions->GetEntries())) continue;

    // Read the BC table and initialize run conditions
    int run = 0;
    ULong64_t globalBC = 0;
    bcTree->SetBranchStatus("*", 0);
    use(bcTree, "fRunNumber", &run);
    use(bcTree, "fGlobalBC", &globalBC);
    std::vector<ULong64_t> bcs(bcTree->GetEntries());
    for (size_t i = 0; i < bcs.size(); ++i) {
      bcTree->GetEntry(i);
      bcs[i] = globalBC;
    }
    bcTree->ResetBranchAddresses();

    if (run != runCached) {
      const auto info = o2::parameters::AggregatedRunInfo::buildAggregatedRunInfo(ccdb.instance(), run);
      const auto timestamp = info.sor / 2 + info.eor / 2;
      o2::base::Propagator::initFieldFromGRP(
        ccdb.getForTimeStamp<o2::parameters::GRPMagField>("GLO/Config/GRPMagField", timestamp));
      if (!o2::base::GeometryManager::isGeometryLoaded())
        ccdb.getForTimeStamp<TGeoManager>("GLO/Config/GeometryAligned", timestamp);
      o2::mch::TrackExtrap::setField();
      auto* lut = o2::base::MatLayerCylSet::rectifyPtrFromFile(
        ccdb.getForTimeStamp<o2::base::MatLayerCylSet>("GLO/Param/MatLUT", timestamp));
      o2::base::Propagator::Instance()->setMatLUT(lut);
      orbitResetUS = ccdb.getSpecific<std::vector<int64_t>>("CTP/Calib/OrbitReset", timestamp)->front();
      runCached = run;
    }

    // Resolve collision BCs and vertices
    std::vector<int> collisionBC(collisions ? collisions->GetEntries() : 0, -1);
    std::vector<int> contributors(collisionBC.size());
    std::vector<std::array<float, 3>> vertices(collisionBC.size(), {0, 0, 0});
    std::vector<std::array<float, 2>> vertexCov(collisionBC.size(), {0, 0});
    int colBC = -1;
    unsigned short nContrib = 0;
    float vx = 0, vy = 0, vz = 0, covXX = 0, covYY = 0;
    if (collisions) {
      collisions->SetBranchStatus("*", 0);
      use(collisions, "fIndexBCs", &colBC);
      use(collisions, "fNumContrib", &nContrib);
      use(collisions, "fPosX", &vx);
      use(collisions, "fPosY", &vy);
      use(collisions, "fPosZ", &vz);
      use(collisions, "fCovXX", &covXX);
      use(collisions, "fCovYY", &covYY);
      for (size_t i = 0; i < collisionBC.size(); ++i) {
        collisions->GetEntry(i);
        collisionBC[i] = colBC;
        contributors[i] = nContrib;
        vertices[i] = {vx, vy, vz};
        vertexCov[i] = {covXX, covYY};
      }
      collisions->ResetBranchAddresses();
    }

    // Assign the first BC in each ambiguous track's compatible BC slice
    std::vector<int> firstBC[2] = {
      std::vector<int>(barrel ? barrel->GetEntries() : 0, -1),
      std::vector<int>(forward ? forward->GetEntries() : 0, -1)
    };
    if (mode == "bc") {
      for (int kind = 0; kind < 2; ++kind) {
        auto* table = kind == 0 ? ambiguousBarrel : ambiguousForward;
        if (!table) continue;
        int trackRow = -1, slice[2] = {-1, -1};
        table->SetBranchStatus("*", 0);
        use(table, kind == 0 ? "fIndexTracks" : "fIndexFwdTracks", &trackRow);
        use(table, "fIndexSliceBCs", slice);
        for (Long64_t i = 0; i < table->GetEntries(); ++i) {
          table->GetEntry(i);
          if (trackRow >= 0 && trackRow < int(firstBC[kind].size()) &&
              slice[0] >= 0 && slice[0] < int(bcs.size()))
            firstBC[kind][trackRow] = slice[0];
        }
        table->ResetBranchAddresses();
      }
    }

    // Select tracks using momentum extrapolated to the assigned vertex
    std::map<long long, std::vector<int>> groups[2];
    std::vector<std::array<float, 2>> vertexKinematics[2] = {
      std::vector<std::array<float, 2>>(barrel ? barrel->GetEntries() : 0),
      std::vector<std::array<float, 2>>(forward ? forward->GetEntries() : 0)
    };
    int barrelCollision = -1;
    float bx = 0, by = 0, bz = 0, alpha = 0, snp = 0, btgl = 0, bqpt = 0;
    unsigned its = 0, flags = 0;
    unsigned char tpc = 0, trd = 0;
    float tof = -1, tofP = -1;
    if (barrel && extra) {
      barrel->SetBranchStatus("*", 0);
      extra->SetBranchStatus("*", 0);
      use(barrel, "fIndexCollisions", &barrelCollision);
      use(barrel, "fX", &bx); use(barrel, "fY", &by); use(barrel, "fZ", &bz);
      use(barrel, "fAlpha", &alpha); use(barrel, "fSnp", &snp);
      use(barrel, "fTgl", &btgl); use(barrel, "fSigned1Pt", &bqpt);
      use(extra, "fFlags", &flags); use(extra, "fITSClusterSizes", &its);
      use(extra, "fTPCNClsFindable", &tpc); use(extra, "fTRDPattern", &trd);
      use(extra, "fTOFChi2", &tof); use(extra, "fTOFExpMom", &tofP);
      for (Long64_t i = 0; i < barrel->GetEntries(); ++i) {
        barrel->GetEntry(i); extra->GetEntry(i);
        const int bcRow = barrelCollision >= 0 ? collisionBC[barrelCollision] : firstBC[0][i];
        if (bcRow < 0 || bqpt == 0) continue;
        const unsigned mask = (its ? kITS : 0) | (tpc ? kTPC : 0) |
                              (trd ? kTRD : 0) | ((tof >= 0 && tofP > 0) ? kTOF : 0);
        if ((mask & required[0]) != required[0] || (mask & forbidden[0]) ||
            (pvOnly && !(flags & 2))) continue;

        const auto vertex = barrelCollision >= 0 ? vertices[barrelCollision] : std::array<float, 3>{0, 0, 0};
        o2::track::TrackParametrization<float> par(bx, alpha, {by, bz, snp, btgl, bqpt});
        if (!o2::base::Propagator::Instance()->propagateToDCABxByBz(
              {vertex[0], vertex[1], vertex[2]}, par, 2.f,
              o2::base::Propagator::MatCorrType::USEMatCorrLUT)) continue;
        const float pt = par.getPt(), eta = par.getEta();
        if (pt < ptMin[0] || pt > ptMax[0] || eta < etaMin[0] || eta > etaMax[0]) continue;

        const long long key = mode == "collision" ? barrelCollision : static_cast<long long>(bcs[bcRow]);
        if (key >= 0) {
          groups[0][key].push_back(i);
          vertexKinematics[0][i] = {pt, eta};
        }
      }
    }

    int forwardCollision = -1;
    unsigned char type = 0, midBits = 0;
    float fx = 0, fy = 0, fz = 0, phi = 0, ftgl = 0, fqpt = 0;
    if (forward) {
      forward->SetBranchStatus("*", 0);
      use(forward, "fIndexCollisions", &forwardCollision);
      use(forward, "fTrackType", &type); use(forward, "fMIDBitMap", &midBits);
      use(forward, "fX", &fx); use(forward, "fY", &fy); use(forward, "fZ", &fz);
      use(forward, "fPhi", &phi); use(forward, "fTgl", &ftgl);
      use(forward, "fSigned1Pt", &fqpt);
      for (Long64_t i = 0; i < forward->GetEntries(); ++i) {
        forward->GetEntry(i);
        if ((type != 3 && type != 4) || fqpt == 0 || ftgl == 0) continue;
        const int bcRow = forwardCollision >= 0 ? collisionBC[forwardCollision] : firstBC[1][i];
        if (bcRow < 0) continue;
        const unsigned mask = kMCH | ((type == 3 || midBits) ? kMID : 0);
        if ((mask & required[1]) != required[1] || (mask & forbidden[1])) continue;

        const auto vertex = forwardCollision >= 0 ? vertices[forwardCollision] : std::array<float, 3>{0, 0, 0};
        const auto cov = forwardCollision >= 0 ? vertexCov[forwardCollision] : std::array<float, 2>{0, 0};
        const double params[5] = {
          fx, std::cos(phi) / ftgl, fy, std::sin(phi) / ftgl,
          fqpt / std::hypot(ftgl, std::sin(phi))
        };
        o2::mch::TrackParam par(fz, params);
        if (!o2::mch::TrackExtrap::extrapToVertex(
              par, vertex[0], vertex[1], vertex[2], cov[0], cov[1])) continue;
        const float pt = std::hypot(par.px(), par.py());
        const float eta = std::asinh(par.pz() / pt);
        if (pt < ptMin[1] || pt > ptMax[1] || eta < etaMin[1] || eta > etaMax[1]) continue;

        const long long key = mode == "collision" ? forwardCollision : static_cast<long long>(bcs[bcRow]);
        if (key >= 0) {
          groups[1][key].push_back(i);
          vertexKinematics[1][i] = {pt, eta};
        }
      }
    }

    // Write one ROOT file per candidate
    std::set<long long> candidateKeys;
    for (const auto& group : groups)
      for (const auto& entry : group) candidateKeys.insert(entry.first);

    for (long long key : candidateKeys) {
      if (selected == nCandidates) break;
      auto& barrelRows = groups[0][key];
      auto& forwardRows = groups[1][key];
      const int counts[3] = {int(barrelRows.size() + forwardRows.size()),
                             int(barrelRows.size()), int(forwardRows.size())};
      bool accepted = true;
      for (int i = 0; i < 3; ++i)
        accepted &= counts[i] >= minCount[i] && counts[i] <= maxCount[i];
      if (!accepted) continue;

      int collision = mode == "collision" ? int(key) : -1;
      if (collision >= 0 &&
          (contributors[collision] < minContrib || contributors[collision] > maxContrib)) continue;
      ULong64_t candidateBC = mode == "collision" ? bcs[collisionBC[collision]] : static_cast<ULong64_t>(key);
      if ((bcMin >= 0 && candidateBC < ULong64_t(bcMin)) ||
          (bcMax >= 0 && candidateBC > ULong64_t(bcMax))) continue;

      std::ostringstream filename;
      filename << outputDir << "/candidate_" << std::setw(4) << std::setfill('0') << selected << ".root";
      TFile output(filename.str().c_str(), "RECREATE");
      TTree info("candidate", "Display candidate metadata");
      int candidateId = selected, modeCode = mode == "collision" ? 0 : 1, df = dfIndex;
      int nBarrel = counts[1], nForward = counts[2];
      int contributorsEvent = collision >= 0 ? contributors[collision] : -1;
      ULong64_t timestampMs = static_cast<ULong64_t>((static_cast<long double>(orbitResetUS) +
        static_cast<long double>(candidateBC) * o2::constants::lhc::LHCBunchSpacingNS / 1000.L) / 1000.L);
      std::string dataframeName = dfName;
      info.Branch("candidateId", &candidateId); info.Branch("dataframeName", &dataframeName);
      info.Branch("mode", &modeCode); info.Branch("run", &run); info.Branch("dataframe", &df);
      info.Branch("bc", &candidateBC); info.Branch("timestampMs", &timestampMs);
      info.Branch("collision", &collision); info.Branch("contributors", &contributorsEvent);
      info.Branch("nBarrel", &nBarrel); info.Branch("nForward", &nForward);
      info.Fill();

      TTree tree("tracks", "Track trajectories in ALICE centimeters");
      int trackId = 0, aodTrackRow = 0, trackCollision = 0, bcSource = 0, charge = 0;
      unsigned detectorMask = 0;
      float pt = 0, eta = 0;
      ULong64_t bc = candidateBC;
      std::vector<float> xyz;
      tree.Branch("run", &run); tree.Branch("df", &df); tree.Branch("trackId", &trackId);
      tree.Branch("aodTrackRow", &aodTrackRow); tree.Branch("collision", &trackCollision);
      tree.Branch("bc", &bc); tree.Branch("bcSource", &bcSource);
      tree.Branch("detectorMask", &detectorMask); tree.Branch("charge", &charge);
      tree.Branch("pt", &pt); tree.Branch("eta", &eta); tree.Branch("xyz", &xyz);

      for (int row : barrelRows) {
        barrel->GetEntry(row); extra->GetEntry(row);
        xyz.clear();
        aodTrackRow = row; trackCollision = barrelCollision; bcSource = barrelCollision >= 0 ? 0 : 1;
        detectorMask = (its ? kITS : 0) | (tpc ? kTPC : 0) |
                       (trd ? kTRD : 0) | ((tof >= 0 && tofP > 0) ? kTOF : 0);
        pt = vertexKinematics[0][row][0]; eta = vertexKinematics[0][row][1];
        charge = bqpt > 0 ? 1 : -1;
        sampleBarrel(bx, by, bz, alpha, snp, btgl, bqpt, maxLength, maxChord, xyz);
        if (xyz.size() >= 6) { tree.Fill(); ++trackId; }
      }
      for (int row : forwardRows) {
        forward->GetEntry(row);
        xyz.clear();
        aodTrackRow = row; trackCollision = forwardCollision; bcSource = forwardCollision >= 0 ? 0 : 1;
        detectorMask = kMCH | ((type == 3 || midBits) ? kMID : 0);
        pt = vertexKinematics[1][row][0]; eta = vertexKinematics[1][row][1];
        charge = fqpt > 0 ? 1 : -1;
        sampleForward(fx, fy, fz, phi, ftgl, fqpt, type,
                      trackCollision >= 0 ? vertices[trackCollision] : std::array<float, 3>{0, 0, 0},
                      trackCollision >= 0 ? vertexCov[trackCollision] : std::array<float, 2>{0, 0}, xyz);
        if (xyz.size() >= 6) { tree.Fill(); ++trackId; }
      }

      output.cd();
      info.Write();
      tree.Write();
      output.Close();
      std::cout << filename.str() << "  BC=" << candidateBC << "  tracks=" << trackId << '\n';
      ++selected;
    }
    for (auto* tree : {barrel, extra, forward})
      if (tree) tree->ResetBranchAddresses();
  }
  std::cout << "Exported " << selected << " / " << nCandidates << " candidate files\n";
}
