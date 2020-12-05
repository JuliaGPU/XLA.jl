using Pkg, JSON, HTTP, Tar, Artifacts

const Artifacts_toml = joinpath(@__DIR__, "../Artifacts.toml")
const libtpu_tag = "libtpu_20201204_RC00"

let libtpu_hash = Artifacts.artifact_hash("libtpu", Artifacts_toml)
    if !Artifacts.artifact_exists(libtpu_hash)
        new_hash = Pkg.Artifacts.create_artifact() do path
            creds = JSON.parse(read(pipeline(`echo $("gcr.io/cloud-tpu-v2-images/libtpu:$libtpu_tag")`,`/usr/bin/docker-credential-gcr get`), String))

            manifest = JSON.parse(String(HTTP.get("https://$(creds["Username"]):$(creds["Secret"])@gcr.io/v2/cloud-tpu-v2-images/libtpu/manifests/$libtpu_tag").body))

            blobsum = manifest["fsLayers"][1]["blobSum"]

            blob = HTTP.get("https://$(creds["Username"]):$(creds["Secret"])@gcr.io/v2/cloud-tpu-v2-images/libtpu/blobs/$blobsum")

            #Tar.extract(IOBuffer(blob.body), path)
            mktemp() do tmp, f
                write(f, blob.body)
                close(f)
                run(`tar -C $path -xf $tmp`)
            end
        end
        if new_hash !== libtpu_hash
            error("Download of libtpu docker image did not yield artifact with correct hash.")
        end
    end
end
