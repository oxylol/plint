/**
 * A release's SHA256SUMS and manifest.json, for whichever of the systems' archives are in the folder
 * (each made by its own job: macos/release.sh, linux/build.sh --package):
 *
 *   CHIMERA-macos.zip       a CHIMERA folder: CLAP, VST3, AU and a how-to
 *   CHIMERA-linux.tar.gz    a CHIMERA folder: CLAP, VST3, the licences and a how-to
 *
 * manifest.json is what an installer needs (public/install.sh, or another tool): the version, each
 * archive's address and checksum, and where each plugin in it goes. `schema` goes up only when a field
 * changes meaning.
 *
 *   bun crates/chimera-plugin/release.ts DIR               the addresses are the GitHub release plugin-v<version>'s
 *   bun crates/chimera-plugin/release.ts DIR --base URL    ... URL's instead (the install tests use file://)
 *   bun crates/chimera-plugin/release.ts DIR --tag TAG     and fail unless TAG is plugin-v<version>
 */
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const repo = "jamesondh/chimera-vocal-synth";
const args = process.argv.slice(2);
const opt = (name: string) => {
  const i = args.indexOf(name);
  return i < 0 ? undefined : args.splice(i, 2)[1];
};
const baseArg = opt("--base");
const tagArg = opt("--tag");
const dir = args[0];
if (!dir || args.length > 1) {
  console.error("usage: bun crates/chimera-plugin/release.ts DIR [--base URL] [--tag TAG]");
  process.exit(2);
}

const cargo = readFileSync(join(import.meta.dir, "../../Cargo.toml"), "utf8");
const version = /\[workspace\.package\][^[]*?\nversion = "([^"]+)"/.exec(cargo)?.[1];
if (!version) throw new Error("no version in Cargo.toml's [workspace.package]");
const tag = `plugin-v${version}`;
if (tagArg !== undefined && tagArg !== tag) {
  console.error(`the tag ${tagArg} doesn't match the version in Cargo.toml (${version}): tag ${tag}, or change the version`);
  process.exit(1);
}
const base = baseArg ?? `https://github.com/${repo}/releases/download/${tag}`;

const systems = {
  macos: {
    file: "CHIMERA-macos.zip",
    minimum_system: "11.0",
    architectures: ["arm64", "x86_64"],
    bundles: [
      { format: "clap", path: "CHIMERA/CHIMERA.clap", install: "~/Library/Audio/Plug-Ins/CLAP" },
      { format: "vst3", path: "CHIMERA/CHIMERA.vst3", install: "~/Library/Audio/Plug-Ins/VST3" },
      { format: "au", path: "CHIMERA/CHIMERA.component", install: "~/Library/Audio/Plug-Ins/Components" },
    ],
  },
  linux: {
    file: "CHIMERA-linux.tar.gz",
    // the newest glibc symbol the build needs: linux/ci.sh fails a build that needs a newer one
    minimum_glibc: "2.35",
    architectures: ["x86_64"],
    bundles: [
      { format: "clap", path: "CHIMERA/CHIMERA.clap", install: "~/.clap" },
      { format: "vst3", path: "CHIMERA/CHIMERA.vst3", install: "~/.vst3" },
    ],
  },
};

const platforms: Record<string, object> = {};
const sums: string[] = [];
for (const [system, { file, ...rest }] of Object.entries(systems)) {
  const path = join(dir, file);
  if (!existsSync(path)) continue;
  const sha256 = new Bun.CryptoHasher("sha256").update(readFileSync(path)).digest("hex");
  sums.push(`${sha256}  ${file}\n`);
  platforms[system] = { url: `${base}/${file}`, sha256, ...rest };
}
if (sums.length === 0) throw new Error(`no ${Object.values(systems).map((s) => s.file).join(" or ")} in ${dir}`);

const manifest = { schema: 1, name: "CHIMERA", id: "com.github.jamesondh.chimera", version, homepage: "https://chimera.audio/plugin/", platforms };
writeFileSync(join(dir, "SHA256SUMS"), sums.join(""));
writeFileSync(join(dir, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n");
console.log(`manifest.json for ${version}: ${Object.keys(platforms).join(", ")}`);
