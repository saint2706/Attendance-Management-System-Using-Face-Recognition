# Security Policy

## Supported Versions

We actively maintain and provide security updates for the following versions:

| Version | Supported          | Status |
| ------- | ------------------ | ------ |
| 1.7.x   | :white_check_mark: | Current stable release |
| 1.6.x   | :white_check_mark: | Security fixes only |
| < 1.6   | :x:                | No longer supported |

**Recommendation**: Always use the latest stable release for the best security posture.

---

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please report it responsibly.

### How to Report

**DO NOT** open a public GitHub issue for security vulnerabilities.

Instead, please use **GitHub's Private Security Reporting**:
1. Go to the [Security Advisories](https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/security/advisories) page
2. Click "Report a vulnerability"
3. Fill out the form with details

### What to Include

When reporting a vulnerability, please provide:

- **Description**: Clear explanation of the vulnerability
- **Impact**: What an attacker could achieve
- **Steps to Reproduce**: Detailed steps to reproduce the issue
- **Proof of Concept**: Code, screenshots, or logs (if available)
- **Affected Versions**: Which versions are vulnerable
- **Suggested Fix**: Your proposed solution (optional but appreciated)
- **Your Contact Info**: Email for follow-up questions

### What to Expect

1. **Acknowledgment**: We'll respond within **48 hours** acknowledging receipt
2. **Assessment**: We'll investigate and confirm the vulnerability within **7 days**
3. **Fix Development**: Critical fixes prioritized, patch developed
4. **Disclosure**: We'll coordinate disclosure timeline with you
5. **Credit**: You'll be credited in the security advisory (unless you prefer anonymity)

### Response Timeline

| Severity | Response Time | Patch Release |
|----------|---------------|---------------|
| **Critical** | 24 hours | 1-3 days |
| **High** | 48 hours | 3-7 days |
| **Medium** | 7 days | 2-4 weeks |
| **Low** | 14 days | Next release |

---

## Security Best Practices

### For Deployment

See our comprehensive [Security Guide](docs/SECURITY.md) for detailed hardening instructions.

**Quick Checklist:**
- ✅ Use strong, unique secrets (never use defaults)
- ✅ Enable HTTPS/TLS with valid certificates
- ✅ Restrict `DJANGO_ALLOWED_HOSTS` to your domains
- ✅ Use PostgreSQL (not SQLite) in production
- ✅ Enable face data encryption (`ENABLE_ENCRYPTION=true`)
- ✅ Configure rate limiting and CORS appropriately
- ✅ Keep dependencies updated
- ✅ Monitor logs and enable Sentry for error tracking

### For Development

**Never commit:**
- Secrets, API keys, or passwords
- `.env` files with real credentials
- Database files or face recognition data
- Private keys or certificates

**Always:**
- Use the provided `.env.example` as a template
- Run `pre-commit install` to catch issues before commit
- Review security implications of your changes
- Include security considerations in PR descriptions

---

## Known Security Considerations

⚠️ **Important**: This system is designed for **attendance tracking**, not high-security access control.

### Face Recognition Limitations

**Known Constraints:**
- Liveness detection helps but doesn't eliminate all spoofing risks
- ~95% accuracy under ideal conditions (varies by demographics)
- Performance depends heavily on lighting, camera quality, and positioning

**Recommendations:**
- Use as **one factor** among multiple authentication methods
- Implement additional verification for sensitive operations
- Configure appropriate thresholds based on your security requirements

See [Fairness & Limitations](docs/FAIRNESS_AND_LIMITATIONS.md) for detailed analysis.

### Data Privacy

**Face Data Handling:**
- Biometric data is sensitive and requires explicit consent
- Encrypted embeddings stored on-premise (never sent externally)
- Comply with GDPR, CCPA, BIPA and local biometric privacy laws

**Your Responsibilities:**
- Obtain informed consent before collecting face data
- Provide clear privacy policy to users
- Implement data access and deletion procedures
- Define and enforce retention policies

See [Data Card](docs/DATA_CARD.md) for comprehensive data handling documentation.

---

## Security Updates

### Subscribing to Alerts

To receive security notifications:

1. **Watch this repository** → "Custom" → Check "Security alerts"
2. **Monitor** [Security Advisories](https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/security)
3. **Review** [CHANGELOG.md](CHANGELOG.md) for security fixes in releases

### Security Fixes

Security patches are clearly marked in release notes with severity levels.

---

## Additional Resources

- **[Security Guide](docs/SECURITY.md)** - Comprehensive hardening guide
- **[Data Card](docs/DATA_CARD.md)** - Data handling and privacy
- **[Configuration Guide](docs/CONFIGURATION.md)** - Secure configuration reference
- **[Fairness & Limitations](docs/FAIRNESS_AND_LIMITATIONS.md)** - Bias and accuracy analysis

---

## Questions?

For security questions (not vulnerabilities):
- Review the [Security Guide](docs/SECURITY.md)
- Ask in [GitHub Discussions](https://github.com/saint2706/Attendance-Management-System-Using-Face-Recognition/discussions)

For actual vulnerabilities, use the private reporting process above.

---

*Last Updated: December 2025*