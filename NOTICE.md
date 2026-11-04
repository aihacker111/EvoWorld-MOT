# Attribution and distribution

The supplied detector YAML and compact StandardJDEReID implementation originate from the user's Falcon-MOT checkout. Its Apache-2.0 LICENSE is retained. The ReID file is reformatted and unused imports removed; substantive architecture is preserved. Source paths and original SHA256 values are recorded in `vendor/falcon_manifest.json`.

The supplied detector checkpoint is copied for this user's local experiment. Its presence does not establish a public redistribution license for the weights or their training data.

Ultralytics is an external dependency, not vendored source; its distribution includes AGPL-3.0 licensing and other available licensing options. TrackEval is optional and has its own upstream license. Other dependencies retain their respective licenses. Retaining Falcon's license does not replace dependency terms. Review all applicable upstream licenses before distributing a combined project or checkpoints.
