#!/bin/bash

# AXF Bot 0 - Secrets Management Script
# Handles environment variables and secrets securely

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to generate a secure secret key
generate_secret_key() {
    print_status "Generating secure secret key..."
    if command -v openssl >/dev/null 2>&1; then
        openssl rand -hex 32
    elif command -v python3 >/dev/null 2>&1; then
        python3 -c "import secrets; print(secrets.token_hex(32))"
    else
        print_error "No suitable method found to generate secret key"
        exit 1
    fi
}

# Function to create .env file from template
create_env_file() {
    local env_type=${1:-development}
    local env_file=".env"
    local template_file="env.${env_type}"
    
    print_status "Creating .env file from ${template_file}..."
    
    if [ ! -f "$template_file" ]; then
        print_error "Template file $template_file not found"
        exit 1
    fi
    
    if [ -f "$env_file" ]; then
        print_warning ".env file already exists. Creating backup..."
        cp "$env_file" "${env_file}.backup.$(date +%Y%m%d_%H%M%S)"
    fi
    
    cp "$template_file" "$env_file"
    
    # Generate secret key if not set
    if grep -q "your-secret-key-change-in-production" "$env_file"; then
        print_status "Generating new secret key..."
        local secret_key=$(generate_secret_key)
        sed -i.bak "s/your-secret-key-change-in-production/$secret_key/g" "$env_file"
        rm "${env_file}.bak"
    fi
    
    print_success ".env file created successfully"
}

# Function to validate environment variables
validate_env() {
    local env_file=${1:-.env}
    
    print_status "Validating environment variables in $env_file..."
    
    if [ ! -f "$env_file" ]; then
        print_error "Environment file $env_file not found"
        exit 1
    fi
    
    # Required variables
    local required_vars=(
        "DATABASE_URL"
        "REDIS_URL"
        "SECRET_KEY"
    )
    
    local missing_vars=()
    
    for var in "${required_vars[@]}"; do
        if ! grep -q "^${var}=" "$env_file" || grep -q "^${var}=$" "$env_file"; then
            missing_vars+=("$var")
        fi
    done
    
    if [ ${#missing_vars[@]} -gt 0 ]; then
        print_error "Missing required environment variables:"
        for var in "${missing_vars[@]}"; do
            echo "  - $var"
        done
        exit 1
    fi
    
    print_success "Environment variables validation passed"
}

# Function to encrypt sensitive values
encrypt_secrets() {
    local env_file=${1:-.env}
    local encrypted_file="${env_file}.encrypted"
    
    print_status "Encrypting sensitive values in $env_file..."
    
    if [ ! -f "$env_file" ]; then
        print_error "Environment file $env_file not found"
        exit 1
    fi
    
    # Sensitive variables to encrypt
    local sensitive_vars=(
        "SECRET_KEY"
        "DB_PASSWORD"
        "REDIS_PASSWORD"
        "FANO_API_KEY"
        "NEWS_API_KEY"
        "ECONOMIC_CALENDAR_API_KEY"
    )
    
    cp "$env_file" "$encrypted_file"
    
    for var in "${sensitive_vars[@]}"; do
        if grep -q "^${var}=" "$encrypted_file"; then
            local value=$(grep "^${var}=" "$encrypted_file" | cut -d'=' -f2-)
            if [ -n "$value" ] && [ "$value" != "your_${var,,}_here" ]; then
                # Simple base64 encoding (in production, use proper encryption)
                local encrypted_value=$(echo -n "$value" | base64)
                sed -i.bak "s/^${var}=.*/${var}=ENC:${encrypted_value}/" "$encrypted_file"
            fi
        fi
    done
    
    rm "${encrypted_file}.bak"
    print_success "Encrypted secrets saved to $encrypted_file"
}

# Function to decrypt sensitive values
decrypt_secrets() {
    local encrypted_file=${1:-.env.encrypted}
    local env_file=${2:-.env}
    
    print_status "Decrypting sensitive values from $encrypted_file..."
    
    if [ ! -f "$encrypted_file" ]; then
        print_error "Encrypted file $encrypted_file not found"
        exit 1
    fi
    
    cp "$encrypted_file" "$env_file"
    
    # Decrypt ENC: prefixed values
    sed -i.bak 's/^\([^=]*\)=ENC:\(.*\)/\1=\2/' "$env_file"
    
    # Decode base64 values
    while IFS= read -r line; do
        if [[ $line == *"=ENC:"* ]]; then
            local var=$(echo "$line" | cut -d'=' -f1)
            local encrypted_value=$(echo "$line" | cut -d'=' -f2- | sed 's/ENC://')
            local decrypted_value=$(echo "$encrypted_value" | base64 -d)
            sed -i.bak "s/^${var}=ENC:.*/${var}=${decrypted_value}/" "$env_file"
        fi
    done < "$env_file"
    
    rm "${env_file}.bak"
    print_success "Decrypted secrets saved to $env_file"
}

# Function to show environment status
show_env_status() {
    local env_file=${1:-.env}
    
    print_status "Environment Status for $env_file"
    echo "=================================="
    
    if [ ! -f "$env_file" ]; then
        print_error "Environment file $env_file not found"
        return 1
    fi
    
    echo "File: $env_file"
    echo "Size: $(wc -c < "$env_file") bytes"
    echo "Lines: $(wc -l < "$env_file")"
    echo ""
    
    echo "Environment Variables:"
    echo "---------------------"
    
    # Show non-sensitive variables
    grep -v "^#" "$env_file" | grep -v "^$" | while IFS= read -r line; do
        local var=$(echo "$line" | cut -d'=' -f1)
        local value=$(echo "$line" | cut -d'=' -f2-)
        
        # Mask sensitive values
        if [[ $var =~ (PASSWORD|SECRET|KEY|TOKEN) ]]; then
            if [ ${#value} -gt 8 ]; then
                value="${value:0:4}...${value: -4}"
            else
                value="***"
            fi
        fi
        
        echo "  $var=$value"
    done
}

# Function to show help
show_help() {
    echo "AXF Bot 0 - Secrets Management Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  create [type]    - Create .env file from template (development|production)"
    echo "  validate [file]  - Validate environment variables"
    echo "  encrypt [file]   - Encrypt sensitive values in .env file"
    echo "  decrypt [file]   - Decrypt sensitive values from encrypted file"
    echo "  status [file]    - Show environment status"
    echo "  generate-key     - Generate a secure secret key"
    echo "  help             - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 create development"
    echo "  $0 validate .env"
    echo "  $0 encrypt .env"
    echo "  $0 status .env"
}

# Main script logic
case "${1:-help}" in
    create)
        create_env_file "$2"
        ;;
    validate)
        validate_env "$2"
        ;;
    encrypt)
        encrypt_secrets "$2"
        ;;
    decrypt)
        decrypt_secrets "$2" "$3"
        ;;
    status)
        show_env_status "$2"
        ;;
    generate-key)
        generate_secret_key
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
