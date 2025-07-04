// Copyright (c) 2025 Sonic Operations Ltd
//
// Use of this software is governed by the Business Source License included
// in the LICENSE file and at soniclabs.com/bsl11.
//
// Change Date: 2028-4-16
//
// On the date above, in accordance with the Business Source License, use of
// this software will be governed by the GNU Lesser General Public License v3.

pipeline {
    agent { label 'pr' }

    options {
        timestamps ()
        timeout(time: 2, unit: 'HOURS')
    }

    environment {
        GOMEMLIMIT = '5GiB'
        CC = 'clang-14'
        CXX = 'clang++-14'
    }

    stages {
        stage('Validate commit') {
            steps {
                script {
                    def CHANGE_REPO = sh (script: "basename -s .git `git config --get remote.origin.url`", returnStdout: true).trim()
                    build job: '/Utils/Validate-Git-Commit', parameters: [
                        string(name: 'Repo', value: "${CHANGE_REPO}"),
                        string(name: 'Branch', value: "${env.CHANGE_BRANCH}"),
                        string(name: 'Commit', value: "${GIT_COMMIT}")
                    ]
                }
            }
        }

        stage('Run tests') {
            stages {
                stage('Check license headers') {
                    steps {
                        sh 'cd scripts/license && ./add_license_header.sh --check'
                    }
                }

                stage('Check Go sources formatting') {
                    steps {
                        sh 'cd go && diff=`gofmt -s -d .` && echo "$diff" && test -z "$diff"'
                    }
                }

                stage('Check C++ sources formatting') {
                    steps {
                        sh 'find cpp/ -iname *.h -o -iname *.cc | xargs clang-format --dry-run -Werror '
                    }
                }

                stage('Build C++ libraries') {
                    steps {
                        sh 'git submodule update --init --recursive'
                        sh 'cd go/lib && ./build_libcarmen.sh'
                    }
                }

                stage('Build Go') {
                    steps {
                        sh 'cd go && go build -v ./...'
                    }
                }

                stage('Run Go tests') {
                    environment {
                        CODECOV_TOKEN = credentials('codecov-uploader-0xsoniclabs-global')
                    }
                    steps {
                        sh 'cd go && go test ./... -coverprofile=coverage.txt -parallel 1 -timeout 60m'
                        sh ('codecov upload-process -r 0xsoniclabs/carmen -f ./go/coverage.txt -t ${CODECOV_TOKEN}')
                    }
                }

                stage('Run Mpt Go Stress Test') {
                    steps {
                        sh 'cd go && go run ./database/mpt/tool stress-test --num-blocks 2000'
                    }
                }

                stage('Run C++ tests') {
                    steps {
                        sh 'cd cpp && bazel test --test_output=errors //...'
                    }
                }
            }
        }
    }
}
