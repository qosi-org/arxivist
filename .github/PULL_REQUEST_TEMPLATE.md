## Summary

What does this PR change and why?

## Type of change

- [ ] Bug fix — incorrect skill behaviour or schema error
- [ ] Skill improvement — better prompting, tighter instructions, new validation
- [ ] Schema change — added/removed/renamed fields in a JSON schema
- [ ] Documentation — docs, examples, README
- [ ] CI / tooling

## Checklist

- [ ] CI passes (`validate-schemas` and `lint-markdown` jobs are green)
- [ ] JSON files are valid (`python3 -m json.tool file.json > /dev/null` exits 0)
- [ ] Schema changes are reflected in templates and the affected agent `.md` files
- [ ] If an agent file was changed: it still has Input Contract, Output Contract, Must NOT Do, and Output Checklist sections
- [ ] `skill/SKILL.md` is still under 500 lines
- [ ] New example SIRs validate against `sir_schema.json`
- [ ] No secrets, API keys, or personal data in any committed file

## Testing

How did you verify this change works? (e.g., ran ArXivist against paper X, checked generated output)

## Related issues

Closes #
