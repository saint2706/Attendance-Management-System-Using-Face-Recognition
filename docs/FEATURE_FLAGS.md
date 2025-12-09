# Feature Flags Configuration Guide

This document explains the feature flag system that allows you to enable or disable advanced features to reduce complexity and maintenance burden.

## Overview

The Attendance Management System includes many advanced features that are valuable for enterprise deployments but may be overwhelming for basic installations. The feature flag system allows you to:

- Choose from pre-configured profiles (basic, standard, advanced)
- Selectively enable/disable individual features
- Reduce code maintenance for optional features
- Scale your deployment as needs grow

## Feature Profiles

### Basic Profile (`FEATURE_PROFILE=basic`)

**Target**: Small deployments, demos, simple attendance tracking (<50 employees)

**Enabled Features:**

- ✅ Core face recognition
- ✅ Basic attendance logging

**Disabled Features:**

- ❌ Liveness detection (relies on user behavior)
- ❌ DeepFace anti-spoofing
- ❌ Scheduled model evaluations
- ❌ Fairness audits
- ❌ Performance profiling (Silk)
- ❌ Encryption (warnings shown in production)
- ❌ Sentry error tracking

**Use When:**

- Running a quick demo or proof-of-concept
- Small office with trusted employees (<50 people)
- Testing or development environments
- Budget/resource constraints

### Standard Profile (`FEATURE_PROFILE=standard`) **[RECOMMENDED]**

**Target**: Most production deployments

**Enabled Features:**

- ✅ Core face recognition
- ✅ Basic attendance logging
- ✅ Motion-based liveness detection
- ✅ Face data encryption at rest
- ✅ Sentry error tracking

**Disabled Features:**

- ❌ DeepFace anti-spoofing (motion-based liveness is sufficient)
- ❌ Scheduled model evaluations (run manually as needed)
- ❌ Fairness audits (run manually)
- ❌ Liveness evaluations (run manually)
- ❌ Performance profiling

**Use When:**

- Production deployment for most organizations
- Security-conscious environments
- 50-500 employees
- Want balance between security and maintenance

### Advanced Profile (`FEATURE_PROFILE=advanced`)

**Target**: Enterprise deployments with compliance requirements

**Enabled Features:**

- ✅ All features enabled
- ✅ Automated monitoring and evaluations
- ✅ Comprehensive auditing
- ✅ Full encryption
- ✅ Performance profiling
- ✅ Both liveness detection types

**Use When:**

- Large enterprise deployment (500+ employees)
- Regulatory compliance requirements (GDPR, CCPA, etc.)
- Security is critical
- Dedicated IT/DevOps resources available
- Budget for infrastructure (Celery, Redis, monitoring)

## Configuration

### Using Profiles

Set the `FEATURE_PROFILE` environment variable in your `.env` file:

```bash
# For basic deployments
FEATURE_PROFILE=basic

# For standard deployments (recommended)
FEATURE_PROFILE=standard

# For advanced deployments
FEATURE_PROFILE=advanced
```

If no profile is set, the system defaults to `advanced` for backward compatibility.

### Overriding Individual Features

You can override specific features regardless of your profile:

```bash
# Use standard profile but enable scheduled evaluations
FEATURE_PROFILE=standard
ENABLE_SCHEDULED_EVALUATIONS=true

# Use advanced profile but disable performance profiling
FEATURE_PROFILE=advanced
ENABLE_PERFORMANCE_PROFILING=false
```

## Available Feature Flags

| Environment Variable | Description | Default (Advanced) |
|---------------------|-------------|-------------------|
| `ENABLE_LIVENESS_DETECTION` | Motion-based anti-spoofing | ✅ true |
| `ENABLE_DEEPFACE_ANTISPOOFING` | DeepFace's anti-spoofing model | ✅ true |
| `ENABLE_SCHEDULED_EVALUATIONS` | Automated nightly model evaluation | ✅ true |
| `ENABLE_FAIRNESS_AUDITS` | Scheduled fairness monitoring | ✅ true |
| `ENABLE_LIVENESS_EVALUATIONS` | Liveness detection evaluation pipeline | ✅ true |
| `ENABLE_PERFORMANCE_PROFILING` | Silk profiling middleware | ✅ true |
| `ENABLE_ENCRYPTION` | Face data encryption at rest | ✅ true |
| `ENABLE_SENTRY` | Error tracking and monitoring | ✅ true |

## Viewing Current Configuration

Use the management command to see which features are currently enabled:

```bash
python manage.py show_features
```

Output example:

```text
=== Feature Flags Configuration ===
Active Profile: STANDARD

  → Recommended for most production deployments

=== Feature Status ===
Motion-based Liveness Detection............ ✓ Enabled
DeepFace Anti-Spoofing..................... ✗ Disabled
Automated Model Evaluations................ ✗ Disabled
Scheduled Fairness Audits.................. ✗ Disabled
Liveness Detection Evaluations............. ✗ Disabled
Performance Profiling (Silk)............... ✗ Disabled
Face Data Encryption....................... ✓ Enabled
Sentry Error Tracking...................... ✓ Enabled
```

## Deployment Recommendations

### Small Office (< 50 employees)

```bash
FEATURE_PROFILE=basic
ENABLE_LIVENESS_DETECTION=true  # Recommended override
ENABLE_ENCRYPTION=true          # Recommended override
```

### Medium Organization (50-500 employees)

```bash
FEATURE_PROFILE=standard
# No overrides needed
```

### Large Enterprise (500+ employees)

```bash
FEATURE_PROFILE=advanced
# All features enabled by default
```

### High-Security Environment

```bash
FEATURE_PROFILE=advanced
# Consider running fairness audits more frequently
```

## Feature Details

### Liveness Detection

**Impact**: Prevents photo/screen spoofing attacks  
**Complexity**: Low  
**Recommendation**: Enable for production even in basic profile

### DeepFace Anti-Spoofing

**Impact**: Additional layer of spoof protection  
**Complexity**: Medium (slower recognition)  
**Recommendation**: Only needed for high-security environments; motion-based liveness is sufficient for most

### Scheduled Evaluations

**Impact**: Automated model quality monitoring  
**Complexity**: High (requires Celery, Redis)  
**Recommendation**: Run manually via `python manage.py eval` unless you have dedicated DevOps

### Fairness Audits

**Impact**: Monitors for demographic bias  
**Complexity**: High (requires Celery, analysis pipeline)  
**Recommendation**: Run manually via `python manage.py fairness_audit` periodically

### Performance Profiling

**Impact**: Database query and request profiling  
**Complexity**: Medium (adds middleware overhead)  
**Recommendation**: Disable in production, enable temporarily for debugging

### Encryption

**Impact**: Encrypts face embeddings at rest  
**Complexity**: Low (requires key management)  
**Recommendation**: Always enable for production deployments

### Sentry

**Impact**: Error tracking and monitoring  
**Complexity**: Low (requires Sentry account)  
**Recommendation**: Highly recommended for production

## Migration Guide

### Existing Deployments

If you're migrating from a version without feature flags:

1. **No action required** - All features remain enabled by default
2. **Optional**: Add `FEATURE_PROFILE=advanced` to `.env` to make this explicit
3. **To reduce complexity**: Switch to `standard` profile when comfortable

### New Deployments

1. **Start with standard profile**: `FEATURE_PROFILE=standard`
2. **Test thoroughly** in your environment
3. **Enable additional features** as needed based on security requirements

## Troubleshooting

### "Feature not working despite being enabled"

Check that all dependencies are installed:

- Celery tasks require Redis and Celery workers running
- Silk requires database migrations
- Sentry requires SENTRY_DSN environment variable

### "Want to disable encryption temporarily"

```bash
# Development only! Not for production
ENABLE_ENCRYPTION=false
DJANGO_DEBUG=true
```

Note: The system will warn about disabled encryption in production environments.

### "How to test different profiles?"

```bash
# Test basic profile
export FEATURE_PROFILE=basic
python manage.py show_features

# Test standard profile  
export FEATURE_PROFILE=standard
python manage.py show_features

# Start server with chosen profile
python manage.py runserver
```

## Performance Impact

| Feature | CPU Impact | Memory Impact | Storage Impact |
|---------|-----------|---------------|----------------|
| Liveness Detection | Low | Low | None |
| DeepFace Anti-Spoofing | Medium | Medium | None |
| Scheduled Evaluations | High (periodic) | Low | Medium (reports) |
| Fairness Audits | High (periodic) | Low | Medium (reports) |
| Performance Profiling | Low-Medium | Low | Medium (profiles) |
| Encryption | Low | Low | None |
| Sentry | Low | Low | None |

## Support

For questions or issues with feature flags:

1. Run `python manage.py show_features` to verify current configuration
2. Check environment variables are set correctly
3. Review logs for feature-related warnings
4. Consult the [Quick Start Guide](QUICKSTART.md) and [Deployment Guide](DEPLOYMENT.md) for deployment guidance
